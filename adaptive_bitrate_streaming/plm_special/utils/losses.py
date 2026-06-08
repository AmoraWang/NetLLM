import torch
import torch.nn.functional as F


def info_nce_contrastive_loss(query, positive, negatives, temperature=0.07):
    """
    单正样本 InfoNCE。query / positive: (N, D)；negatives: (N, K, D)。

    标签恒为 0（正样本在 logits 第 0 列）。
    """
    t = max(float(temperature), 1e-6)
    q = F.normalize(query.float(), dim=-1)
    p = F.normalize(positive.float(), dim=-1)
    n = F.normalize(negatives.float(), dim=-1)
    pos_logit = (q * p).sum(dim=-1, keepdim=True) / t
    neg_logit = torch.bmm(n, q.unsqueeze(-1)).squeeze(-1) / t
    logits = torch.cat([pos_logit, neg_logit], dim=-1)
    targets = torch.zeros(logits.size(0), dtype=torch.long, device=logits.device)
    return F.cross_entropy(logits, targets)


def knowledge_distillation_kl(student_logits, teacher_logits, temperature=2.0):
    """
    KL( teacher || student ) on bitrate classes, averaged over batch and time.

    :param student_logits: (B, C, T)
    :param teacher_logits: (B, C, T) raw logits or unnormalized scores
    """
    t = max(float(temperature), 1e-6)
    student_log_probs = F.log_softmax(student_logits / t, dim=1)
    teacher_probs = F.softmax(teacher_logits / t, dim=1)
    return F.kl_div(student_log_probs, teacher_probs, reduction='batchmean') * (t ** 2)


def compute_abr_train_loss(
    actions_pred,
    labels,
    loss_fn,
    loss_type='ce',
    teacher_logits=None,
    kd_alpha=0.5,
    kd_temperature=2.0,
    teacher_is_prob=False,
    align_loss=None,
    align_lambda=0.0,
    bw_vae_kl=None,
    bw_vae_kl_lambda=0.0,
):
    """
    :param actions_pred: (B, BITRATE_LEVELS, T) student logits
    :param labels: (B, T) hard bitrate indices
    :return: (loss_tensor, loss_info_dict with float scalars)
    """
    loss_hard = loss_fn(actions_pred, labels)
    info = {
        'loss_hard': float(loss_hard.detach().cpu()),
        'loss_soft': 0.0,
        'loss_align': 0.0,
        'loss_bw_vae_kl': 0.0,
    }

    def _add_aux(base_loss):
        total = base_loss
        if align_loss is not None and align_lambda > 0:
            lam = float(align_lambda)
            total = total + lam * align_loss
            info['loss_align'] = float(align_loss.detach().cpu())
        if bw_vae_kl is not None and bw_vae_kl_lambda > 0:
            lam_bw = float(bw_vae_kl_lambda)
            total = total + lam_bw * bw_vae_kl
            info['loss_bw_vae_kl'] = float(bw_vae_kl.detach().cpu())
        info['loss_total'] = float(total.detach().cpu())
        return total

    def _add_align(base_loss):
        return _add_aux(base_loss)

    if loss_type == 'ce':
        return _add_align(loss_hard), info

    if loss_type != 'ce_kl':
        raise ValueError(f"Unknown loss_type: {loss_type!r}. Choose 'ce' or 'ce_kl'.")

    if teacher_logits is None:
        raise ValueError("loss_type='ce_kl' requires teacher_logits in the batch.")

    if teacher_is_prob:
        t = max(float(kd_temperature), 1e-6)
        student_log_probs = F.log_softmax(actions_pred / t, dim=1)
        teacher_probs = teacher_logits.clamp(1e-6, 1.0 - 1e-6)
        teacher_probs = teacher_probs / teacher_probs.sum(dim=1, keepdim=True)
        loss_soft = F.kl_div(student_log_probs, teacher_probs, reduction='batchmean') * (t ** 2)
    else:
        loss_soft = knowledge_distillation_kl(
            actions_pred, teacher_logits, temperature=kd_temperature
        )

    alpha = float(kd_alpha)
    loss_total = alpha * loss_hard + (1.0 - alpha) * loss_soft
    info['loss_soft'] = float(loss_soft.detach().cpu())
    return _add_align(loss_total), info
