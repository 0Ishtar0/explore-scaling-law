import torch
from torch import nn

from data import TrainCurve


class MPL(nn.Module):

    def __init__(self):
        super().__init__()
        self.L0 = nn.Parameter(torch.tensor(1.0, dtype=torch.float32))
        self.A = nn.Parameter(torch.tensor(1.0, dtype=torch.float32))
        self.alpha = nn.Parameter(torch.tensor(1.0, dtype=torch.float32))
        self.B = nn.Parameter(torch.tensor(1.0, dtype=torch.float32))
        self.C = nn.Parameter(torch.tensor(1.0, dtype=torch.float32))
        self.beta = nn.Parameter(torch.tensor(1.0, dtype=torch.float32))
        self.gamma = nn.Parameter(torch.tensor(1.0, dtype=torch.float32))

    def forward(self, data: TrainCurve):
        S1 = data.S1
        lrs = data.learning_rates
        lr_sum = data.lr_sum
        step = data.steps
        lr_gap = data.lr_gap
        device = lrs.device
        dtype = torch.float32
        if step.numel() == 0:
            LD = torch.empty(0, device=device, dtype=dtype)
        elif torch.all(step == 0):
            LD = torch.zeros_like(step, dtype=dtype)
        else:
            max_s_val_in_batch = torch.max(step)
            if max_s_val_in_batch == 0:
                LD = torch.zeros_like(step, dtype=dtype)
            else:
                k_indices = torch.arange(1, max_s_val_in_batch.item() + 1,
                                         device=device, dtype=torch.long)
                lrs_k = lrs[k_indices]
                lr_gap_k = lr_gap[k_indices]
                lr_sum_k_minus_1 = lr_sum[k_indices - 1]
                s_values = step.unsqueeze(1)
                lr_sum_at_s = lr_sum[step].unsqueeze(1)
                delta_lr_sum = lr_sum_at_s - lr_sum_k_minus_1.unsqueeze(0)

                term_C_lrs_delta = self.C * (lrs_k.unsqueeze(0) ** (-self.gamma)) * delta_lr_sum
                base_pow = 1 + term_C_lrs_delta
                clamped_base_pow = torch.clamp(base_pow, min=1e-10)
                powered_term = clamped_base_pow ** (-self.beta)
                summand = lr_gap_k.unsqueeze(0) * (1 - powered_term)
                mask = k_indices.unsqueeze(0) <= s_values
                masked_summand = summand * mask

                LD = torch.sum(masked_summand, dim=1)

        pred = self.L0 + self.A * S1 ** (-self.alpha) + self.B * LD
        return pred


class LRA(nn.Module):

    def __init__(self):
        super().__init__()
        self.L0 = nn.Parameter(torch.tensor(1.0, dtype=torch.float32))
        self.A = nn.Parameter(torch.tensor(1.0, dtype=torch.float32))
        self.alpha = nn.Parameter(torch.tensor(1.0, dtype=torch.float32))
        self.B = nn.Parameter(torch.tensor(1.0, dtype=torch.float32))
        self.beta = nn.Parameter(torch.tensor(1.0, dtype=torch.float32))
        self.C = nn.Parameter(torch.tensor(1.0, dtype=torch.float32))
        self.gamma = nn.Parameter(torch.tensor(1.0, dtype=torch.float32))

    def forward(self, data: TrainCurve):
        S1 = data.S1
        N = data.N
        pred = self.L0 + self.A * S1 ** (- self.alpha) + self.B * \
            N ** (-self.beta) - self.C * N ** (-self.gamma)
        return pred
