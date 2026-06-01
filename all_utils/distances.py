import torch
import torch.nn.functional as F


def wasserstein_from_logits_3d_fast(logits1: torch.Tensor, logits2: torch.Tensor) -> torch.Tensor:
    """
    Calculates the 1D Wasserstein Distance for each token in a sequence using a
    fast, vectorized PyTorch implementation.

    Args:
        logits1 (torch.Tensor): Logits from model 1. Shape: (batch_size, seq_len, vocab_size).
        logits2 (torch.Tensor): Logits from model 2. Shape: (batch_size, seq_len, vocab_size).

    Returns:
        torch.Tensor: A tensor of Wasserstein distances. Shape: (batch_size, seq_len).
    """
    # 1. Convert logits to probabilities
    p1 = F.softmax(logits1, dim=-1)
    p2 = F.softmax(logits2, dim=-1)

    # 2. Calculate the cumulative distribution functions (CDFs)
    # The cumsum operation is the key to vectorizing this
    cdf1 = torch.cumsum(p1, dim=-1)
    cdf2 = torch.cumsum(p2, dim=-1)

    # 3. Calculate the Wasserstein-1 distance
    # The distance is the sum of the absolute differences between the CDFs
    # We slice off the last element of the CDF, as it's always 1 and not needed for the sum.
    w_distance = torch.sum(torch.abs(cdf1[..., :-1] - cdf2[..., :-1]), dim=-1)

    return w_distance


def jsd_from_logits_3d(logits1: torch.Tensor, logits2: torch.Tensor, reduction: str = 'batchmean') -> torch.Tensor:
    """
    Calculates the JSD between two sets of 3D logits (batch, seq_len, vocab_size).

    Args:
        logits1 (torch.Tensor): Logits from model 1. Shape: (batch_size, seq_len, vocab_size).
        logits2 (torch.Tensor): Logits from model 2. Shape: (batch_size, seq_len, vocab_size).
        reduction (str): 'batchmean' averages the divergence across all token positions.
                         'none' returns element-wise divergence. Default: 'batchmean'.

    Returns:
        torch.Tensor: The JSD value. A scalar if reduction is 'batchmean'.
    """
    # Softmax is correctly applied over the last dimension (vocab_size)
    p1 = F.softmax(logits1, dim=-1)
    p2 = F.softmax(logits2, dim=-1)
    
    log_p1 = F.log_softmax(logits1, dim=-1)
    log_p2 = F.log_softmax(logits2, dim=-1)

    m = 0.5 * (p1 + p2)
    log_m = m.log()

    # F.kl_div with 'batchmean' reduction will average the divergence
    # over all elements, effectively giving the mean JSD per token.
    kl1 = F.kl_div(log_p1, log_m, reduction=reduction, log_target=True)
    kl2 = F.kl_div(log_p2, log_m, reduction=reduction, log_target=True)

    jsd = 0.5 * (kl1 + kl2)
    
    return jsd


def bild_loss(logits_s, logits_t, top_k=8, temperature=3, student_led=False):
    """
    Bi-directional Logits Difference loss.

    Args:
        logits_s (torch.Tensor): the student logits, shape (batch_size, seq_len, vocab_size).
        logits_t (torch.Tensor): the teacher logits, shape (batch_size, seq_len, vocab_size).
        top_k (int, optional): choose top-k logits for calculating loss, defaults to 8.
        temperature (int, optional): the temperature, defaults to 3.
        student_led (bool, optional): if true, calculate student-led logits difference loss (t-LD), else t-LD.
    """
    pair_num = top_k * (top_k-1) // 2

    if not student_led:
        # select top-k teacher logits & corresponding student logits
        with torch.no_grad():
            select_logits_t, select_pos = torch.topk(logits_t, k=top_k, dim=-1)
        select_logits_s = torch.gather(logits_s, 2, select_pos)
    else:
        # select top-k student logits & corresponding teacher logits
        select_logits_s, select_pos = torch.topk(logits_s, k=top_k, dim=-1)
        with torch.no_grad():
            select_logits_t = torch.gather(logits_t, 2, select_pos)

    scaled_logits_t = select_logits_t / temperature
    scaled_logits_s = select_logits_s / temperature

    # calculate logit difference
    def get_prob_diff(logits):
        b, n, v = logits.size()
        i, j = torch.triu_indices(v, v, offset=1)

        logits_diff = logits[..., i] - logits[..., j]

        return logits_diff

    logits_diff_t = get_prob_diff(scaled_logits_t)
    logits_diff_s = get_prob_diff(scaled_logits_s)

    logits_diff_t = F.softmax(logits_diff_t, dim=-1)

    loss = F.kl_div(F.log_softmax(logits_diff_s, dim=-1), logits_diff_t, reduction='none')

    loss = loss.sum(-1, keepdim=True)

    return loss