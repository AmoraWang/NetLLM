#!/usr/bin/env python3
"""
可视化 ABRLLM v2/v3 的 Reprogramming + Alignment（网络语义 ↔ LLM 语义对齐）。

「槽位 (slot)」是什么？
  mapping_layer 把 LLM 全词表 (约 12.8 万 token) 压缩成 tiny_vocab_size=2048 个
  **可学习的虚拟语义锚点**。每个槽位 i 是全词表嵌入的线性组合，可理解为
  「一类与网络状态对齐的 LLM 语义原型」。AlignmentLayer 用交叉注意力让当前
  网络 state 查询这 2048 个锚点，再聚合 Value 得到 LLM 维度的对齐向量。

输出（默认在 --output-dir）：
  - HOW_TO_READ.txt          如何解读图表（中文）
  - attn_heatmap_topK.png    时间步 × Top-K 槽位 注意力热力图
  - attn_per_timestep.png    若干时间步上的槽位注意力条形图
  - slot_lexicon.png         Top 槽位 → 词表 token 翻译（mapping_layer 权重）
  - alignment_dashboard.html 汇总页（热力图 + 带 token 标签的条形图）

示例：
  cd adaptive_bitrate_streaming
  python run/visualize_alignment.py \\
      --model-dir data/ft_plms/llama_base/.../best_model \\
      --exp-pool-path artifacts/exp_pools/eval_trajectories_with_logits.pkl \\
      --sample-index 0 \\
      --output-dir artifacts/alignment_viz/sample0
"""
from __future__ import annotations

import argparse
import html
import os
import pickle
import sys

_ABR_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ABR_ROOT not in sys.path:
    sys.path.insert(0, _ABR_ROOT)

import numpy as np
import torch

from plm_special.data.dataset import ExperienceDataset
from plm_special.utils.utils import set_random_seed
from run.abr_viz_common import build_abrllm, patch_viz_args

try:
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
except ImportError as e:
    raise ImportError("请安装 matplotlib: pip install matplotlib") from e


def _configure_matplotlib_chinese() -> str | None:
    """
    配置 matplotlib 中文字体，避免热力图/标题显示为方框。
    返回实际选用的字体名；若未找到则返回 None（可安装 fonts-wqy-microhei 或 noto-cjk）。
    """
    plt.rcParams["axes.unicode_minus"] = False

    # 先从常见系统路径注册字体文件（WSL/Ubuntu 更可靠）
    font_paths = [
        "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
        "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc",
        "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf",
        "/usr/share/fonts/truetype/arphic/uming.ttc",
        "/usr/share/fonts/truetype/arphic/ukai.ttc",
        "/mnt/c/Windows/Fonts/msyh.ttc",
        "/mnt/c/Windows/Fonts/simhei.ttf",
    ]
    for path in font_paths:
        if os.path.isfile(path):
            try:
                font_manager.fontManager.addfont(path)
                name = font_manager.FontProperties(fname=path).get_name()
                plt.rcParams["font.sans-serif"] = [name, "DejaVu Sans"]
                return name
            except Exception:
                continue

    available = {f.name for f in font_manager.fontManager.ttflist}
    candidates = [
        "WenQuanYi Micro Hei",
        "WenQuanYi Zen Hei",
        "Noto Sans CJK SC",
        "Noto Sans CJK JP",
        "Source Han Sans SC",
        "Source Han Sans CN",
        "SimHei",
        "Microsoft YaHei",
        "PingFang SC",
        "Arial Unicode MS",
        "Droid Sans Fallback",
        "AR PL UMing CN",
        "AR PL UKai CN",
    ]
    for name in candidates:
        if name in available:
            plt.rcParams["font.sans-serif"] = [name, "DejaVu Sans"]
            return name
    for f in font_manager.fontManager.ttflist:
        n = f.name
        if any(
            key in n
            for key in ("CJK", "Hei", "Song", "YaHei", "PingFang", "WenQuanYi", "Noto Sans")
        ):
            plt.rcParams["font.sans-serif"] = [n, "DejaVu Sans"]
            return n
    return None


HOW_TO_READ = """\
=== 如何看懂对齐可视化 ===

1) 槽位 (slot) 是什么？
   - 2048 个「虚拟语义锚点」，由 LLM 全词表经 mapping_layer 压缩得到。
   - 不是某一个真实 token，而是一组 token 嵌入的加权组合（见 slot_lexicon.png）。

2) attn_heatmap_topK.png（核心图）
   - 横轴：按全局平均注意力选出的 Top-K 个槽位（编号 0~2047）。
   - 纵轴：经验窗口内的时间步 t（每个 t 对应一个 ABR chunk 状态）。
   - 颜色越亮：该时刻网络 state 越「关注」该语义槽位。
   - 若对齐有效：不同 t 应有不同亮带；若整图均匀发灰，则查询几乎不选择性。

3) attn_per_timestep.png
   - 单个时间步上，模型最关注的若干槽位及其注意力权重（多头平均后）。

4) slot_lexicon.png / alignment_dashboard.html
   - 对每个高注意力槽位，列出 mapping_layer 权重最大的真实 subword/token。
   - 这是「槽位 → 自然语言」的近似翻译（线性组合里贡献最大的词）。
   - 若 top token 与网络/码率/缓冲相关词共现，说明 reprogramming 学到了可用语义。

5) 注意
   - 请在 eval() 下运行；训练模式 dropout 会扰动注意力。
   - 建议对比不同 sample-index（高缓冲 vs 低缓冲、升码 vs 降码）。
"""


def _unpack_dataset_item(item):
    """ExperienceDataset 在无 teacher_logits 时返回 4 元组，有时返回 5 元组。"""
    if len(item) == 4:
        return item[0], item[1], item[2], item[3]
    if len(item) == 5:
        return item[0], item[1], item[2], item[3]
    raise ValueError(f"ExperienceDataset 样本长度应为 4 或 5，得到 {len(item)}")


def _build_tiny_vocab(model) -> torch.Tensor:
    """(num_slots, llm_dim)"""
    word_embeddings_float = model.word_embeddings.permute(1, 0).to(torch.float32)
    tiny = model.mapping_layer(word_embeddings_float).permute(1, 0)
    return tiny


@torch.no_grad()
def _encode_state_query(model, states, actions, returns):
    """返回 state_emb (B,L,D) 与 tiny_vocab (S,D)。"""
    states = states.to(model.device).float()
    actions = actions.to(model.device).float()
    returns = returns.to(model.device).float()

    action_emb_llm = model.action_embedding(actions)
    return_emb_llm = model.return_embedding(returns)
    action_emb_for_state = model.action_proj_to_state_dim(action_emb_llm)
    return_emb_for_state = model.return_proj_to_state_dim(return_emb_llm)

    encoder_output = model.state_encoder(
        states,
        action_embedding=action_emb_for_state,
        return_embedding=return_emb_for_state,
    )
    if not model.state_use_self_attention:
        raise RuntimeError("可视化脚本当前要求 --state-use-self-attention（与 ABRLLM_v2 默认一致）")
    state_emb, _, _ = encoder_output
    tiny_vocab = _build_tiny_vocab(model)
    return state_emb, tiny_vocab


@torch.no_grad()
def _alignment_attn(model, state_emb, tiny_vocab):
    """state 路径交叉注意力，返回 attn_mean (L,S)。"""
    _, attn = model.alignment_layer(
        state_emb, tiny_vocab, tiny_vocab, return_attn=True
    )
    # attn: (B, H, L, S)
    attn_mean = attn[0].mean(dim=0).cpu().numpy()  # (L, S)
    return attn_mean


def _slot_top_tokens(model, slot_idx: int, top_n: int = 8) -> list[tuple[str, float]]:
    """mapping_layer.weight: (num_slots, vocab_size)"""
    w = model.mapping_layer.weight[slot_idx].detach().cpu()
    vals, ids = torch.topk(w, k=min(top_n, w.numel()))
    out = []
    for v, tid in zip(vals.tolist(), ids.tolist()):
        try:
            text = model.tokenizer.decode([tid], skip_special_tokens=False)
        except Exception:
            text = f"<id_{tid}>"
        text = text.replace("\n", "\\n")
        out.append((text, v))
    return out


def _select_top_slots(attn_ls: np.ndarray, top_k: int) -> np.ndarray:
    """按时间平均注意力选出 top_k 槽位下标。"""
    importance = attn_ls.mean(axis=0)  # (S,)
    k = min(top_k, importance.shape[0])
    return np.argsort(importance)[-k:][::-1]


def _plot_heatmap(attn_ls: np.ndarray, slot_ids: np.ndarray, out_path: str, title: str):
    sub = attn_ls[:, slot_ids]  # (L, K)
    fig, ax = plt.subplots(figsize=(max(8, sub.shape[1] * 0.35), max(4, sub.shape[0] * 0.35)))
    im = ax.imshow(sub.T, aspect="auto", cmap="YlOrRd", origin="lower")
    ax.set_xlabel("时间步 t (窗口内 chunk 索引)")
    ax.set_ylabel("槽位 slot id (Top-K)")
    ax.set_yticks(range(len(slot_ids)))
    ax.set_yticklabels([str(int(s)) for s in slot_ids])
    ax.set_title(title)
    plt.colorbar(im, ax=ax, label="注意力 (多头平均)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _plot_timestep_bars(
    attn_ls: np.ndarray,
    slot_ids: np.ndarray,
    timesteps: list[int],
    model,
    out_path: str,
):
    n = len(timesteps)
    fig, axes = plt.subplots(n, 1, figsize=(10, 2.2 * n), squeeze=False)
    for i, t in enumerate(timesteps):
        ax = axes[i, 0]
        weights = attn_ls[t, slot_ids]
        order = np.argsort(weights)[::-1][: min(12, len(slot_ids))]
        slots = slot_ids[order]
        w = weights[order]
        labels = [f"#{int(s)}" for s in slots]
        ax.barh(range(len(slots)), w, color="steelblue")
        ax.set_yticks(range(len(slots)))
        ax.set_yticklabels(labels)
        ax.invert_yaxis()
        ax.set_xlabel("注意力")
        ax.set_title(f"时间步 t={t}")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _plot_slot_lexicon(model, slot_ids: np.ndarray, attn_scores: np.ndarray, out_path: str, top_tokens: int = 6):
    rows = len(slot_ids)
    fig, ax = plt.subplots(figsize=(12, max(4, 0.45 * rows)))
    ax.axis("off")
    lines = ["槽位翻译 (mapping_layer 权重 Top token)\n"]
    for rank, (sid, score) in enumerate(zip(slot_ids, attn_scores)):
        tokens = _slot_top_tokens(model, int(sid), top_n=top_tokens)
        tok_str = ", ".join([f"{repr(t)}({w:.3g})" for t, w in tokens[:top_tokens]])
        lines.append(f"#{rank+1} slot {int(sid):4d}  mean_attn={score:.4f}  →  {tok_str}\n")
    ax.text(0.01, 0.99, "".join(lines), va="top", ha="left", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _write_dashboard(
    out_dir: str,
    attn_ls: np.ndarray,
    slot_ids: np.ndarray,
    model,
    actions: list,
    heatmap_name: str,
):
    rows_html = []
    importance = attn_ls.mean(axis=0)
    for sid in slot_ids[:16]:
        toks = _slot_top_tokens(model, int(sid), top_n=6)
        tok_html = ", ".join(
            f"<code>{html.escape(t)}</code> <small>({w:.2f})</small>" for t, w in toks
        )
        rows_html.append(
            f"<tr><td>{int(sid)}</td><td>{importance[int(sid)]:.4f}</td><td>{tok_html}</td></tr>"
        )

    L = attn_ls.shape[0]
    ts_show = [0, L // 2, L - 1] if L >= 3 else list(range(L))
    bar_imgs = []
    for t in ts_show:
        w = attn_ls[t, slot_ids[:12]]
        order = np.argsort(w)[::-1]
        fig, ax = plt.subplots(figsize=(9, 2.5))
        labels = []
        for j in order:
            sid = int(slot_ids[j])
            top_tok = _slot_top_tokens(model, sid, top_n=1)
            label = f"#{sid} {top_tok[0][0]!r}" if top_tok else f"#{sid}"
            labels.append(label[:28])
        ax.barh(range(len(order)), w[order], color="coral")
        ax.set_yticks(range(len(order)))
        ax.set_yticklabels(labels, fontsize=7)
        ax.invert_yaxis()
        ax.set_title(f"t={t}  action={actions[t] if t < len(actions) else '?'}")
        fname = f"bar_t{t}.png"
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, fname), dpi=120)
        plt.close(fig)
        bar_imgs.append((t, fname))

    bars_html = "".join(
        f'<h3>时间步 {t}</h3><img src="{html.escape(fn)}" width="900"/><br/>'
        for t, fn in bar_imgs
    )

    page = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>ABR Alignment Dashboard</title></head>
<body>
<h1>ABRLLM 语义对齐可视化</h1>
<pre>{html.escape(HOW_TO_READ)}</pre>
<h2>热力图 (Top 槽位 × 时间)</h2>
<img src="{html.escape(heatmap_name)}" width="1000"/>
<h2>高注意力槽位 → 自然语言 (Top token)</h2>
<table border="1" cellpadding="6">
<tr><th>槽位 id</th><th>平均注意力</th><th>Top tokens (mapping 权重)</th></tr>
{''.join(rows_html)}
</table>
<h2>带 token 标签的逐步注意力</h2>
{bars_html}
</body></html>"""
    with open(os.path.join(out_dir, "alignment_dashboard.html"), "w", encoding="utf-8") as f:
        f.write(page)


def main():
    parser = argparse.ArgumentParser(description="Visualize ABRLLM v2/v3 alignment / reprogramming slots")
    parser.add_argument("--model-dir", required=True, help="best_model 或 checkpoint 目录")
    parser.add_argument("--exp-pool-path", required=True, help="经验池 pickle")
    parser.add_argument("--sample-index", type=int, default=0, help="ExperienceDataset 样本下标")
    parser.add_argument("--output-dir", type=str, default="artifacts/alignment_viz")
    parser.add_argument("--top-k-slots", type=int, default=32, help="热力图显示的槽位数")
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
        help="须与 checkpoint 一致；v3 含 contrast_proj，不可用 v2 类加载 v3 权重",
    )
    parser.add_argument("--contrast-dim", type=int, default=256, help="v3 对比 MLP 输出维，与训练 --contrast-dim 一致")
    parser.add_argument("--align-lambda", type=float, default=0.1, help="仅影响 v3 模型是否构建 contrast_proj（与训练一致即可）")
    parser.add_argument("--state-feature-dim", type=int, default=256)
    parser.add_argument("--state-attn-hidden-dim", type=int, default=2048)
    parser.add_argument("--state-use-self-attention", action="store_true", default=True)
    parser.add_argument("--fusion-method", default="weighted_sum")
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--key-dim", type=int, default=128)
    parser.add_argument("--frozen", action="store_true", default=True)
    parser.add_argument("--device", default=None)
    parser.add_argument("--seed", type=int, default=666)
    args = parser.parse_args()

    patch_viz_args(args)

    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, "HOW_TO_READ.txt"), "w", encoding="utf-8") as f:
        f.write(HOW_TO_READ)

    font_used = _configure_matplotlib_chinese()
    if font_used:
        print(f"matplotlib 中文字体: {font_used}")
    else:
        print(
            "警告: 未检测到系统中文字体，图中中文可能仍为方框。"
            "可安装: sudo apt install fonts-wqy-microhei 或 fonts-noto-cjk"
        )

    set_random_seed(args.seed)

    exp_pool = pickle.load(open(args.exp_pool_path, "rb"))
    dataset = ExperienceDataset(
        exp_pool,
        gamma=args.gamma,
        scale=args.scale,
        max_length=args.w,
        sample_step=args.sample_step,
    )
    if args.sample_index >= len(dataset):
        raise IndexError(f"sample_index {args.sample_index} >= dataset size {len(dataset)}")

    states, actions, returns, timesteps = _unpack_dataset_item(dataset[args.sample_index])
    states_t = torch.tensor(np.array(states), dtype=torch.float32).unsqueeze(0)  # (1, T, 6, 6)
    actions_t = torch.tensor(actions, dtype=torch.float32).reshape(1, -1, 1)
    returns_t = torch.tensor(returns, dtype=torch.float32).reshape(1, -1, 1)

    model = build_abrllm(args, load_weights=True)

    state_emb, tiny_vocab = _encode_state_query(model, states_t, actions_t, returns_t)
    attn_ls = _alignment_attn(model, state_emb, tiny_vocab)  # (L, S)
    slot_ids = _select_top_slots(attn_ls, args.top_k_slots)
    slot_scores = attn_ls.mean(axis=0)[slot_ids]

    heatmap_name = "attn_heatmap_topK.png"
    _plot_heatmap(
        attn_ls,
        slot_ids,
        os.path.join(args.output_dir, heatmap_name),
        title=f"State→TinyVocab 交叉注意力 (Top-{len(slot_ids)} 槽位, 多头平均)",
    )

    L = attn_ls.shape[0]
    ts_show = sorted(set([0, L // 2, max(0, L - 1)]))
    _plot_timestep_bars(
        attn_ls,
        slot_ids,
        ts_show,
        model,
        os.path.join(args.output_dir, "attn_per_timestep.png"),
    )

    _plot_slot_lexicon(
        model,
        slot_ids[: min(20, len(slot_ids))],
        slot_scores[: min(20, len(slot_ids))],
        os.path.join(args.output_dir, "slot_lexicon.png"),
    )

    _write_dashboard(args.output_dir, attn_ls, slot_ids, model, actions, heatmap_name)

    np.savez(
        os.path.join(args.output_dir, "attn_data.npz"),
        attn=attn_ls,
        top_slot_ids=slot_ids,
        actions=np.array(actions),
    )

    print(f"槽位数 (tiny_vocab): {tiny_vocab.shape[0]}")
    print(f"窗口长度 L={attn_ls.shape[0]}, 已保存到: {args.output_dir}")
    print("请阅读 HOW_TO_READ.txt，并在浏览器打开 alignment_dashboard.html")


if __name__ == "__main__":
    main()
