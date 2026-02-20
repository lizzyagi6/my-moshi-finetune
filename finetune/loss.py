import torch
from torch.nn import functional as F


def compute_loss_with_mask(
    logits: torch.Tensor,
    target: torch.Tensor,
    target_mask: torch.Tensor,
    mode: str,
    first_codebook_weight_multiplier: float = 1.0,
    text_padding_weight: float = 1.0,
    text_padding_ids: set[int] | None = None,
):
    target = torch.where(target_mask, target, torch.zeros_like(target))

    weights = target_mask.float() #B x dep_q (8 or 1) x T
    if mode == "audio":
        weights[:, 0] *= first_codebook_weight_multiplier #not this is the first audio code
    elif mode == "text":
        assert text_padding_ids is not None
        for id in text_padding_ids:
            weights[target == id] *= text_padding_weight # e.g. text_padding_weight==0.5

    #logits could be (text) torch.Size([2, 1, 750, 32000])
    #or (audio) torch.Size([2, 8, 750, 2048])
    logits = logits.view(-1, logits.size(-1)).float()
    target = target.view(-1)
    weights = weights.view(-1)
    mb_loss = F.cross_entropy(logits, target, reduction="none")
    mb_loss = torch.where(weights > 0.0, mb_loss * weights, torch.zeros_like(mb_loss))
    mb_loss = torch.sum(mb_loss) / torch.sum(weights)

    return mb_loss
