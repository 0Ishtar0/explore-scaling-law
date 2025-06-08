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
        self.lambd = 0.99
        self.C = nn.Parameter(torch.tensor(1.0, dtype=torch.float32))

    def forward(self, data: TrainCurve):
        S1 = data.S1
        lrs = data.learning_rates
        steps = data.steps
        device = S1.device
        dtype = S1.dtype

        actual_steps_formula = steps + 1

        max_s_formula = torch.max(actual_steps_formula).item()
        X_s_values = lrs[:max_s_formula] - lrs[1:max_s_formula + 1]
        m_arange = torch.arange(max_s_formula, device=device)
        indices_diff = m_arange.unsqueeze(1) - m_arange.unsqueeze(0)
        lambda_powers = self.lambd ** indices_diff
        toeplitz_operator = torch.tril(lambda_powers)
        current_term_prime_values = toeplitz_operator @ X_s_values
        s2_contributions = torch.cumsum(current_term_prime_values, dim=0)
        initial_zero = torch.tensor([0.0], device=device, dtype=dtype)
        S2_all_s = torch.cat((initial_zero, s2_contributions), dim=0)

        S2 = S2_all_s[actual_steps_formula]
        S1 = torch.clamp(S1, min=1e-10)
        term_A = self.A * (S1 ** (-self.alpha))
        term_C = self.C * S2

        pred = self.L0 + term_A - term_C
        return pred

    def forward_low_mem(self, data: TrainCurve):
        S1 = data.S1
        lrs = data.learning_rates

        device = S1.device
        dtype = S1.dtype

        actual_steps_formula = data.steps + 1
        max_s_formula = torch.max(actual_steps_formula).item()

        S2_all_s = torch.zeros(max_s_formula + 1, device=device, dtype=dtype)

        current_term_prime = torch.tensor(0.0, device=device, dtype=dtype)

        if max_s_formula > 0:
            for s_iter in range(1, max_s_formula + 1):
                eta_s_minus_1 = lrs[s_iter - 1]
                eta_s = lrs[s_iter]
                X_s = eta_s_minus_1 - eta_s
                current_term_prime = X_s + self.lambd * current_term_prime
                if s_iter == 1:
                    S2_all_s[s_iter] = current_term_prime
                else:
                    S2_all_s[s_iter] = S2_all_s[s_iter - 1] + current_term_prime

        S2 = S2_all_s[actual_steps_formula]
        s1_clamped = torch.clamp(S1, min=1e-10)
        term_A = self.A * (s1_clamped ** (-self.alpha))
        term_C = self.C * S2

        pred = self.L0 + term_A - term_C
        return pred


class _MPL(nn.Module):

    def __init__(self):
        super().__init__()
        self.L0 = nn.Parameter(torch.tensor(1.0, dtype=torch.float32))
        self.A = nn.Parameter(torch.tensor(1.0, dtype=torch.float32))
        self.alpha = nn.Parameter(torch.tensor(1.0, dtype=torch.float32))
        self.B = 80
        self.C = 1
        self.beta = 1
        self.gamma = 1

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
                # summand = lr_gap_k.unsqueeze(0) * (1 - powered_term)
                # SPL
                summand = lr_gap_k.unsqueeze(0)
                mask = k_indices.unsqueeze(0) <= s_values
                masked_summand = summand * mask

                LD = torch.sum(masked_summand, dim=1)

        pred = self.L0 + self.A * S1 ** (-self.alpha) + self.B * LD
        return pred
