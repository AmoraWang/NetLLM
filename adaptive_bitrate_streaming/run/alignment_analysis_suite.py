#!/usr/bin/env python3
"""
依次执行对齐可视化的四项深化分析：
  1) 多样本对比（升码/降码/高缓冲/低缓冲）
  2) 主槽位遮挡（#879/#859）对 logits 的影响
  3) 训练前（随机初始化）vs 训练后注意力对比
  4) Top-50 token 词云 / 主题词频（槽位→自然语言增强）

示例：
  cd adaptive_bitrate_streaming
  bash bash/run_alignment_suite.sh
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import pickle
import sys
from collections import Counter

_ABR_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ABR_ROOT not in sys.path:
    sys.path.insert(0, _ABR_ROOT)

import numpy as np
import torch
import torch.nn as nn

from config import cfg
from plm_special.data.dataset import ExperienceDataset
from plm_special.utils.utils import set_random_seed

try:
    import matplotlib.pyplot as plt
except ImportError as e:
    raise ImportError("请安装 matplotlib") from e

from run.visualize_alignment import (
    _alignment_attn,
    _build_tiny_vocab,
    _configure_matplotlib_chinese,
    _encode_state_query,
    _plot_heatmap,
    _select_top_slots,
    _slot_top_tokens,
    _unpack_dataset_item,
)

# Pensieve 行 1 = 缓冲（归一化）
BUFFER_ROW = 1


def _json_safe(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    return obj


def _load_args_namespace(cli_args):
    from run.abr_viz_common import patch_viz_args

    patch_viz_args(cli_args)
    return cli_args


def _build_model(cli_args, load_weights=True):
    from run.abr_viz_common import build_abrllm

    return build_abrllm(cli_args, load_weights=load_weights)


def _batch_from_dataset(dataset, sample_index: int):
    item = _unpack_dataset_item(dataset[sample_index])
    states, actions, returns, timesteps = item
    states_t = torch.tensor(np.array(states), dtype=torch.float32).unsqueeze(0)
    actions_t = torch.tensor(actions, dtype=torch.float32).reshape(1, -1, 1)
    returns_t = torch.tensor(returns, dtype=torch.float32).reshape(1, -1, 1)
    timesteps_t = torch.tensor(timesteps, dtype=torch.int32).unsqueeze(0)
    meta = {
        "actions": np.array(actions, dtype=np.int64),
        "buffer_end": float(np.array(states)[-1, BUFFER_ROW, -1]),
        "bitrate_range": (int(np.min(actions)), int(np.max(actions))),
    }
    return states_t, actions_t, returns_t, timesteps_t, meta


def _reset_alignment_modules(model):
    """仅重置对齐相关模块（保留 PLM 预训练权重）。"""
    modules = [
        model.state_encoder,
        model.alignment_layer,
        model.mapping_layer,
        model.action_embedding,
        model.return_embedding,
        model.timestep_embedding,
        model.action_projection,
        model.action_proj_to_state_dim,
        model.return_proj_to_state_dim,
    ]
    if model.state_use_self_attention:
        modules.extend([model.action_proj_to_llm_dim, model.return_proj_to_llm_dim])
    for mod in modules:
        for m in mod.modules():
            if hasattr(m, "reset_parameters"):
                m.reset_parameters()


@torch.no_grad()
def _alignment_attn_masked(model, state_emb, tiny_vocab, mask_slot_ids):
    _, attn = model.alignment_layer(
        state_emb, tiny_vocab, tiny_vocab,
        return_attn=True, mask_slot_ids=mask_slot_ids,
    )
    return attn[0].mean(dim=0).cpu().numpy()


@torch.no_grad()
def _full_forward_last_logits(model, states_t, actions_t, returns_t, timesteps_t, mask_slot_ids=None):
    """完整 forward；可选在 alignment 阶段遮挡槽位。返回最后一帧 6 类 logits。"""
    # 临时 monkey-patch：仅 state 路径 alignment 带 mask
    align = model.alignment_layer
    orig_forward = align.forward

    def patched_forward(q, k, v, return_attn=False, mask_slot_ids=None):
        return orig_forward(q, k, v, return_attn=return_attn, mask_slot_ids=mask_slot_ids)

    align.forward = patched_forward
    try:
        states = states_t.to(model.device).float()
        actions = actions_t.to(model.device).float()
        returns = returns_t.to(model.device).float()
        timesteps = timesteps_t.to(model.device).long()

        word_embeddings_float = model.word_embeddings.permute(1, 0).to(torch.float32)
        tiny_vocab = model.mapping_layer(word_embeddings_float).permute(1, 0)

        action_emb_llm = model.action_embedding(actions)
        return_emb_llm = model.return_embedding(returns)
        action_emb_for_state = model.action_proj_to_state_dim(action_emb_llm)
        return_emb_for_state = model.return_proj_to_state_dim(return_emb_llm)

        encoder_output = model.state_encoder(
            states, action_embedding=action_emb_for_state, return_embedding=return_emb_for_state,
        )
        state_emb, action_emb, return_emb = encoder_output

        state_embeddings = align(
            state_emb, tiny_vocab, tiny_vocab, mask_slot_ids=mask_slot_ids,
        )
        action_embeddings = align(action_emb, tiny_vocab, tiny_vocab)
        return_embeddings = align(return_emb, tiny_vocab, tiny_vocab)

        timestep_embeddings = model.timestep_embedding(timesteps)
        state_embeddings = state_embeddings + timestep_embeddings
        action_embeddings = action_embeddings + timestep_embeddings
        return_embeddings = return_embeddings + timestep_embeddings

        instruction = (
            "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
            "You are a specialized AI assistant for Adaptive Bitrate (ABR) streaming optimization. "
            "You will be provided with a sequence of vector embeddings representing (Reward, State, Action) tuples. "
            "You must interpret these numerical patterns to make real-time bitrate decisions.<|eot_id|>"
            "<|start_header_id|>user<|end_header_id|>\n\n"
            f"CONTEXT: {model.data_description}\n"
            f"OBJECTIVE: {model.task_description}\n"
            "The embedded sequence follows immediately. Predict the optimal next action (0-5).<|eot_id|>"
            "<|start_header_id|>assistant<|end_header_id|>\n\n"
        )
        instruction_ids = model.tokenizer(
            instruction, return_tensors="pt", padding=True, truncation=True, max_length=2048,
        ).input_ids.to(model.device).long()
        instruction_embeddings = model.plm.get_input_embeddings()(instruction_ids)

        concated = [instruction_embeddings]
        seq_len = states.shape[1]
        for i in range(seq_len):
            concated.append(return_embeddings[:, i:i + 1, :])
            concated.append(state_embeddings[:, i:i + 1, :])
            concated.append(action_embeddings[:, i:i + 1, :])
        concated_embeddings = torch.cat(concated, dim=1)
        attention_mask = torch.ones(
            (concated_embeddings.shape[0], concated_embeddings.shape[1]),
            dtype=torch.long, device=model.device,
        )
        out = model.plm(inputs_embeds=concated_embeddings, attention_mask=attention_mask, output_hidden_states=True)
        if isinstance(out, dict):
            hidden = out["hidden_states"][-1]
        else:
            hidden = out.hidden_states[-1]
        instruction_len = instruction_embeddings.shape[1]
        state_positions = [instruction_len + 1 + i * 3 for i in range(seq_len)]
        state_outputs = hidden[:, state_positions, :]
        action_pred = model.action_projection(state_outputs)
        return action_pred[0, -1, :].detach().cpu().numpy()
    finally:
        align.forward = orig_forward


def _find_diverse_samples(dataset, max_scan: int | None = None):
    """扫描 ExperienceDataset，挑选四类窗口。"""
    n = len(dataset)
    scan_n = n if max_scan is None else min(n, max_scan)
    records = []
    for idx in range(scan_n):
        states, actions, returns, _ = _unpack_dataset_item(dataset[idx])
        actions = np.asarray(actions, dtype=np.int64)
        diff = np.diff(actions) if len(actions) > 1 else np.array([0])
        records.append({
            "idx": idx,
            "max_up": int(diff.max()) if diff.size else 0,
            "max_down": int(diff.min()) if diff.size else 0,
            "n_unique_actions": int(len(np.unique(actions))),
            "buffer_end": float(np.asarray(states)[-1, BUFFER_ROW, -1]),
            "action_last": int(actions[-1]),
        })

    def _pick(name, key_fn, reverse=True):
        sorted_recs = sorted(records, key=key_fn, reverse=reverse)
        chosen = sorted_recs[0]["idx"]
        return name, chosen, sorted_recs[0]

    picks = {}
    picks["bitrate_up"] = _pick("bitrate_up", lambda r: (r["max_up"], r["n_unique_actions"]))
    picks["bitrate_down"] = _pick("bitrate_down", lambda r: (-r["max_down"], r["n_unique_actions"]))
    picks["high_buffer"] = _pick("high_buffer", lambda r: r["buffer_end"])
    picks["low_buffer"] = _pick("low_buffer", lambda r: r["buffer_end"], reverse=False)
    # 若与 sample0 同类，尽量去重
    used = set()
    final = {}
    for key, (name, idx, rec) in picks.items():
        if idx not in used:
            final[key] = (idx, rec)
            used.add(idx)
        else:
            for alt in sorted(records, key=lambda r: (0 if r["idx"] in used else 1, -r["buffer_end"])):
                if alt["idx"] not in used:
                    final[key] = (alt["idx"], alt)
                    used.add(alt["idx"])
                    break
    return final


def step1_multi_sample(model, dataset, out_dir, top_k_slots, w):
    print("\n=== [1/4] 多样本注意力对比 ===")
    step_dir = os.path.join(out_dir, "step1_multi_sample")
    os.makedirs(step_dir, exist_ok=True)

    picks = _find_diverse_samples(dataset, max_scan=3000)
    summary = {}
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    for ax, (key, (idx, rec)) in zip(axes, picks.items()):
        states_t, actions_t, returns_t, _, meta = _batch_from_dataset(dataset, idx)
        state_emb, tiny_vocab = _encode_state_query(model, states_t, actions_t, returns_t)
        attn = _alignment_attn(model, state_emb, tiny_vocab)
        top_slots = _select_top_slots(attn, min(16, top_k_slots))
        sub = attn[:, top_slots]
        im = ax.imshow(sub.T, aspect="auto", cmap="YlOrRd", origin="lower")
        ax.set_title(
            f"{key}\nidx={idx} buf={meta['buffer_end']:.2f} "
            f"br={meta['bitrate_range']} up={rec['max_up']} down={rec['max_down']}"
        )
        ax.set_xlabel("t")
        ax.set_ylabel("slot")
        plt.colorbar(im, ax=ax, fraction=0.046)
        _plot_heatmap(
            attn, _select_top_slots(attn, top_k_slots),
            os.path.join(step_dir, f"heatmap_{key}_idx{idx}.png"),
            title=f"{key} (dataset idx={idx})",
        )
        summary[key] = {
            "dataset_index": int(idx),
            "meta": meta,
            "record": rec,
            "top5_slots": [int(s) for s in _select_top_slots(attn, 5)],
            "top1_share": float(attn.max(axis=1).mean()),
        }
        print(f"  {key}: idx={idx}, top5 slots={summary[key]['top5_slots']}")

    fig.suptitle("四类样本 Top-16 槽位注意力对比", fontsize=14)
    fig.tight_layout()
    fig.savefig(os.path.join(step_dir, "multi_sample_grid.png"), dpi=150)
    plt.close(fig)

    with open(os.path.join(step_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(_json_safe(summary), f, indent=2, ensure_ascii=False)
    return picks, summary


def step2_ablation(model, dataset, out_dir, sample_index, dominant_slots):
    print("\n=== [2/4] 主槽位遮挡实验 ===")
    step_dir = os.path.join(out_dir, "step2_ablation")
    os.makedirs(step_dir, exist_ok=True)

    states_t, actions_t, returns_t, timesteps_t, meta = _batch_from_dataset(dataset, sample_index)
    masks = {
        "none": None,
        "mask_top2": [int(dominant_slots[0]), int(dominant_slots[1])],
    }
    results = {}
    for name, mask in masks.items():
        logits = _full_forward_last_logits(
            model, states_t, actions_t, returns_t, timesteps_t, mask_slot_ids=mask,
        )
        pred = int(logits.argmax())
        results[name] = {
            "logits": logits.tolist(),
            "pred_bitrate": pred,
            "label_action": int(meta["actions"][-1]),
        }
        print(f"  {name}: pred={pred}, label={meta['actions'][-1]}, logits={np.round(logits, 3)}")

    # 仅 alignment 输出上的投影（不经过 PLM）
    state_emb, tiny_vocab = _encode_state_query(model, states_t, actions_t, returns_t)
    align_only = {}
    for name, mask in masks.items():
        out = model.alignment_layer(
            state_emb, tiny_vocab, tiny_vocab, mask_slot_ids=mask,
        )
        logits_h = model.action_projection(out[:, -1:, :])[0, 0].detach().cpu().numpy()
        align_only[name] = {"logits_head_only": logits_h.tolist(), "pred": int(logits_h.argmax())}
        print(f"  [head-only] {name}: pred={align_only[name]['pred']}")

    fig, ax = plt.subplots(figsize=(8, 4))
    x = np.arange(6)
    w = 0.35
    ax.bar(x - w / 2, results["none"]["logits"], width=w, label="无遮挡 (full)")
    ax.bar(x + w / 2, results["mask_top2"]["logits"], width=w, label=f"遮挡 {masks['mask_top2']}")
    ax.set_xticks(x)
    ax.set_xlabel("码率档位")
    ax.set_ylabel("logit")
    ax.set_title(f"最后一帧 logits 对比 (sample {sample_index})")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(step_dir, "ablation_logits.png"), dpi=150)
    plt.close(fig)

    report = {"sample_index": sample_index, "mask_slots": masks["mask_top2"], "full_forward": results, "head_only": align_only}
    with open(os.path.join(step_dir, "report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)


def step3_random_vs_trained(cli_args, dataset, out_dir, sample_index, top_k_slots):
    print("\n=== [3/4] 随机初始化 vs 训练后 ===")
    step_dir = os.path.join(out_dir, "step3_init_vs_trained")
    os.makedirs(step_dir, exist_ok=True)

    states_t, actions_t, returns_t, _, _ = _batch_from_dataset(dataset, sample_index)

    model_trained = _build_model(cli_args, load_weights=True)
    state_emb, tiny_vocab = _encode_state_query(model_trained, states_t, actions_t, returns_t)
    attn_trained = _alignment_attn(model_trained, state_emb, tiny_vocab)

    model_rand = _build_model(cli_args, load_weights=False)
    _reset_alignment_modules(model_rand)
    state_emb_r, tiny_vocab_r = _encode_state_query(model_rand, states_t, actions_t, returns_t)
    attn_random = _alignment_attn(model_rand, state_emb_r, tiny_vocab_r)

    slots = _select_top_slots(attn_trained, top_k_slots)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, attn, title in zip(
        axes,
        [attn_random, attn_trained],
        ["随机初始化 alignment", "训练后 alignment"],
    ):
        im = ax.imshow(attn[:, slots].T, aspect="auto", cmap="YlOrRd", origin="lower")
        ax.set_title(title)
        ax.set_xlabel("t")
        ax.set_ylabel("slot")
        plt.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(os.path.join(step_dir, "compare_heatmap.png"), dpi=150)
    plt.close(fig)

    def _entropy_mean(a):
        p = a / a.sum(axis=1, keepdims=True)
        return float(-(p * np.log(p + 1e-12)).sum(axis=1).mean())
    stats = {
        "trained_top1_mean": float(attn_trained.max(axis=1).mean()),
        "random_top1_mean": float(attn_random.max(axis=1).mean()),
        "trained_entropy_mean": _entropy_mean(attn_trained),
        "random_entropy_mean": _entropy_mean(attn_random),
        "cosine_mean_trained_vs_random": float(
            np.mean([
                (a @ b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9)
                for a, b in zip(attn_trained, attn_random)
            ])
        ),
    }
    with open(os.path.join(step_dir, "stats.json"), "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)
    print(f"  trained top1={stats['trained_top1_mean']:.4f}, random top1={stats['random_top1_mean']:.4f}")
    print(f"  entropy trained={stats['trained_entropy_mean']:.2f}, random={stats['random_entropy_mean']:.2f}")

    del model_trained, model_rand
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _slot_token_counter(model, slot_idx: int, top_n: int = 50) -> Counter:
    w = model.mapping_layer.weight[slot_idx].detach().cpu()
    vals, ids = torch.topk(w, k=min(top_n, w.numel()))
    c = Counter()
    for v, tid in zip(vals.tolist(), ids.tolist()):
        try:
            text = model.tokenizer.decode([tid], skip_special_tokens=True).strip()
        except Exception:
            text = ""
        if not text:
            text = f"id{tid}"
        c[text] += max(v, 0.0)
    return c


def step4_wordcloud_lexicon(model, out_dir, slot_ids, top_tokens: int = 50):
    print("\n=== [4/4] Top-50 token 词云 / 词频 ===")
    step_dir = os.path.join(out_dir, "step4_lexicon_enhanced")
    os.makedirs(step_dir, exist_ok=True)

    slot_ids = [int(s) for s in slot_ids[:8]]
    all_text = []
    per_slot = {}
    for sid in slot_ids:
        cnt = _slot_token_counter(model, sid, top_n=top_tokens)
        per_slot[sid] = cnt.most_common(20)
        for tok, w in cnt.items():
            all_text.extend([tok] * max(1, int(w * 1000)))

    # 词频条形图（不依赖 wordcloud）
    fig, axes = plt.subplots(len(slot_ids), 1, figsize=(10, 2.2 * len(slot_ids)), squeeze=False)
    for i, sid in enumerate(slot_ids):
        ax = axes[i, 0]
        items = per_slot[sid][:15]
        labels = [t[:24] for t, _ in items]
        weights = [w for _, w in items]
        ax.barh(range(len(labels)), weights, color="teal")
        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels(labels, fontsize=7)
        ax.invert_yaxis()
        ax.set_title(f"slot {sid} — mapping top-{top_tokens} 权重词频")
    fig.tight_layout()
    fig.savefig(os.path.join(step_dir, "token_freq_bars.png"), dpi=150)
    plt.close(fig)

    try:
        from wordcloud import WordCloud

        wc = WordCloud(width=1200, height=600, background_color="white").generate(" ".join(all_text))
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.imshow(wc, interpolation="bilinear")
        ax.axis("off")
        ax.set_title("Top 槽位合并词云 (权重加权)")
        fig.tight_layout()
        fig.savefig(os.path.join(step_dir, "wordcloud_merged.png"), dpi=150)
        plt.close(fig)
        for sid in slot_ids[:4]:
            wc_s = WordCloud(width=800, height=400, background_color="white").generate(
                " ".join([t for t, w in _slot_token_counter(model, sid, top_tokens).items() for _ in range(max(1, int(w * 500)))])
            )
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.imshow(wc_s, interpolation="bilinear")
            ax.axis("off")
            ax.set_title(f"slot {sid} 词云")
            fig.savefig(os.path.join(step_dir, f"wordcloud_slot_{sid}.png"), dpi=150)
            plt.close(fig)
        print("  已生成 wordcloud（需 wordcloud 包）")
    except ImportError:
        print("  未安装 wordcloud，已跳过词云图（仍保留 token_freq_bars.png）")

    with open(os.path.join(step_dir, "slot_top_tokens.json"), "w", encoding="utf-8") as f:
        json.dump({str(k): v for k, v in per_slot.items()}, f, indent=2, ensure_ascii=False)


def _write_suite_report(out_dir, summary1, dominant_slots):
    path = os.path.join(out_dir, "ANALYSIS_REPORT.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write("# 对齐分析套件结果解读\n\n")
        f.write("## 1. 多样本对比 (step1_multi_sample/)\n\n")
        f.write("- 查看 `multi_sample_grid.png` 与各类 `heatmap_*.png`。\n")
        f.write("- 若升码/降码/高缓冲/低缓冲的 **top 槽位或亮带分布明显不同**，说明对齐随网络状态变化。\n")
        f.write("- 详情见 `step1_multi_sample/summary.json`。\n\n")
        f.write("```json\n")
        f.write(json.dumps(_json_safe(summary1), indent=2, ensure_ascii=False))
        f.write("\n```\n\n")
        f.write("## 2. 遮挡实验 (step2_ablation/)\n\n")
        f.write(f"- 遮挡主槽位 `{dominant_slots[:2]}` 后对比 `ablation_logits.png` 与 `report.json`。\n")
        f.write("- 若 logits/预测档位变化大，说明主槽位对决策有实质贡献。\n\n")
        f.write("## 3. 随机 vs 训练 (step3_init_vs_trained/)\n\n")
        f.write("- `compare_heatmap.png`：左随机初始化、右训练后。\n")
        f.write("- 训练后应更尖锐、模式更稳定；若两者相似，可能结构来自 PLM 而非对齐训练。\n\n")
        f.write("## 4. 增强词表 (step4_lexicon_enhanced/)\n\n")
        f.write("- `token_freq_bars.png` / `wordcloud_*.png`：槽位对应的 top-50 token 分布。\n")
    print(f"\n报告已写入: {path}")


def main():
    parser = argparse.ArgumentParser(description="ABR alignment analysis suite (4 steps)")
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--exp-pool-path", required=True)
    parser.add_argument("--output-dir", default="artifacts/alignment_viz/suite")
    parser.add_argument("--sample-index", type=int, default=0, help="遮挡实验用的样本")
    parser.add_argument("--top-k-slots", type=int, default=32)
    parser.add_argument("--plm-type", default="llama")
    parser.add_argument("--plm-size", default="base")
    parser.add_argument("--rank", type=int, default=128)
    parser.add_argument("--w", type=int, default=20)
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--scale", type=int, default=1000)
    parser.add_argument("--sample-step", type=int, default=None)
    parser.add_argument(
        "--abr-llm-version",
        choices=("v2", "v3"),
        default="v2",
        help="须与 checkpoint 一致",
    )
    parser.add_argument("--contrast-dim", type=int, default=256)
    parser.add_argument("--align-lambda", type=float, default=0.1)
    parser.add_argument("--state-feature-dim", type=int, default=256)
    parser.add_argument("--state-attn-hidden-dim", type=int, default=2048)
    parser.add_argument("--state-use-self-attention", action="store_true", default=True)
    parser.add_argument("--fusion-method", default="weighted_sum")
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--key-dim", type=int, default=128)
    parser.add_argument("--frozen", action="store_true", default=True)
    parser.add_argument("--device", default=None)
    parser.add_argument("--seed", type=int, default=666)
    parser.add_argument("--scan-limit", type=int, default=3000, help="多样本搜索扫描的 dataset 上限")
    args = parser.parse_args()

    args = _load_args_namespace(args)
    os.makedirs(args.output_dir, exist_ok=True)
    _configure_matplotlib_chinese()
    set_random_seed(args.seed)

    exp_pool = pickle.load(open(args.exp_pool_path, "rb"))
    dataset = ExperienceDataset(
        exp_pool, gamma=args.gamma, scale=args.scale,
        max_length=args.w, sample_step=args.sample_step,
    )

    print("加载训练后模型…")
    model = _build_model(args, load_weights=True)

    picks, summary1 = step1_multi_sample(model, dataset, args.output_dir, args.top_k_slots, args.w)

    states_t, actions_t, returns_t, timesteps_t, _ = _batch_from_dataset(dataset, args.sample_index)
    state_emb, tiny_vocab = _encode_state_query(model, states_t, actions_t, returns_t)
    attn0 = _alignment_attn(model, state_emb, tiny_vocab)
    dominant = _select_top_slots(attn0, 5)

    step2_ablation(model, dataset, args.output_dir, args.sample_index, dominant)
    step3_random_vs_trained(args, dataset, args.output_dir, args.sample_index, args.top_k_slots)
    step4_wordcloud_lexicon(model, args.output_dir, dominant, top_tokens=50)

    _write_suite_report(args.output_dir, summary1, dominant)
    print(f"\n全部完成，输出目录: {args.output_dir}")


if __name__ == "__main__":
    main()
