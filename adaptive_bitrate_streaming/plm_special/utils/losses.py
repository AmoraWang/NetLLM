import torch.nn.functional as F


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
    }

    if loss_type == 'ce':
        info['loss_total'] = info['loss_hard']
        return loss_hard, info

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
    info['loss_total'] = float(loss_total.detach().cpu())
    return loss_total, info
