import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Callable

class PCLayer(nn.Module):
    def __init__(
        self,
        T: int = 10,
        local_learning_rate: float = 1e-4,
        energy_fn: Optional[Callable] = None,
        x_init: Optional[Callable] = None,
        is_holding_error: bool = False,
        update_bias: bool = True
    ):
        super().__init__()
        self.T = T
        self.local_lr = local_learning_rate
        self.energy_fn = energy_fn
        self.x_init = x_init if x_init else lambda b, s, d: torch.randn(b, s, d) * 0.1
        self.is_holding_error = is_holding_error
        self.update_bias = update_bias
        self.clamp_value = 1.0
        self.weight_clamp = 5.0
        self.input_ids = None
        self.position_ids = None
        self.x = None
        self.layers = {}
        self.kind = None
        self._x_cache = {}
        self._W_cache = {}
        self._energy = None
        self._errors = []

    def init_x(
        self,
        batch_size: int,
        seq_len: int,
        layer: Optional[nn.Module] = None,
        q_proj: Optional[nn.Linear] = None,
        k_proj: Optional[nn.Linear] = None,
        v_proj: Optional[nn.Linear] = None,
        kind: str = "fc1",
        input_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None
    ):
        B, S = batch_size, seq_len
        self.kind = kind
        self.input_ids = input_ids
        self.position_ids = position_ids
        self.layers = {
            "main": layer,
            "q": q_proj,
            "k": k_proj,
            "v": v_proj
        } if kind == "attn" else {"main": layer} if kind in ("fc1", "fc2", "output_attn", "final_output") else layer

        if kind == "embed":
            assert input_ids is not None and position_ids is not None, "input_ids and position_ids required for embed"
            W_word = self.layers["word"].weight
            W_pos = self.layers["pos"].weight
            x_word = W_word[input_ids] * 0.1
            x_pos = W_pos[position_ids] * 0.1
            self.x = {"word": x_word, "pos": x_pos}
            self._x_cache["word"] = x_word.detach()
            self._x_cache["pos"] = x_pos.detach()
        elif kind in ("fc1", "fc2", "output_attn", "final_output"):
            assert isinstance(self.layers["main"], nn.Linear), f"{kind} requires nn.Linear"
            W = self.layers["main"].weight
            x_dim = W.shape[1]
            self.x = self.x_init(B, S, x_dim)
            self._x_cache[kind] = self.x.detach()
        elif kind == "attn":
            assert all(p in self.layers for p in ("q", "k", "v")), "Q, K, V projections required"
            H_out = self.layers["v"].weight.shape[0]
            self.x = self.x_init(B, S, H_out)
            self._x_cache[kind] = self.x.detach()
            if self._x_cache[kind] is None:
                raise ValueError(f"Failed to initialize x for attn: x_init returned None")
        else:
            raise ValueError(f"Unsupported kind: {kind}")

    def compute_mu(self):
        if self.kind == "embed":
            mu = self.x["word"] + self.x["pos"]
        elif self.kind in ("fc1", "fc2", "output_attn", "final_output"):
            mu = self.x @ self.layers["main"].weight.T
            if self.layers["main"].bias is not None:
                mu += self.layers["main"].bias
            if self.kind == "fc1":
                mu = F.gelu(mu)
        elif self.kind == "attn":
            Q = self.layers["q"](self.x)
            K = self.layers["k"](self.x)
            V = self.layers["v"](self.x)
            scores = Q @ K.transpose(-2, -1) / math.sqrt(Q.size(-1))
            mask = torch.tril(torch.ones(self.x.shape[1], self.x.shape[1], device=self.x.device, dtype=torch.bool))
            mask = mask.unsqueeze(0).expand(self.x.shape[0], -1, -1)
            scores = scores.masked_fill(mask == 0, float('-inf'))
            attn_weights = scores.softmax(dim=-1)
            mu = attn_weights @ V
        else:
            raise ValueError(f"Unsupported kind: {self.kind}")

        if torch.any(torch.isnan(mu)) or torch.any(torch.isinf(mu)):
            print(f"Warning: NaN or inf in mu for {self.kind}")
            print(f"x: {torch.any(torch.isnan(self.x)).item()}, {torch.any(torch.isinf(self.x)).item()}")
            if self.kind in ("fc1", "fc2", "output_attn", "final_output"):
                print(f"weight: {torch.any(torch.isnan(self.layers['main'].weight)).item()}, "
                      f"{torch.any(torch.isinf(self.layers['main'].weight)).item()}")
            elif self.kind == "attn":
                print(f"attn_weights: {torch.any(torch.isnan(attn_weights)).item()}, "
                      f"{torch.any(torch.isinf(attn_weights)).item()}")
        return mu

    def step(self, target_activity: torch.Tensor):
        if target_activity is None:
            raise ValueError(f"Target activity is None for {self.kind}")
        mu = self.compute_mu()
        if target_activity.shape != mu.shape:
            raise ValueError(f"Target shape {target_activity.shape} does not match mu shape {mu.shape} for {self.kind}")

        error = target_activity - mu
        if torch.any(torch.isnan(error)) or torch.any(torch.isinf(error)):
            print(f"Warning: NaN or inf in error for {self.kind}")
            print(f"mu: {torch.any(torch.isnan(mu)).item()}, {torch.any(torch.isinf(mu)).item()}")
            print(f"target: {torch.any(torch.isnan(target_activity)).item()}, "
                  f"{torch.any(torch.isinf(target_activity)).item()}")

        max_grad_norm = 0.5
        if self.kind == "embed":
            grad_word = -self.local_lr * error
            grad_pos = -self.local_lr * error
            grad_word = torch.clamp(grad_word, -max_grad_norm, max_grad_norm)
            grad_pos = torch.clamp(grad_pos, -max_grad_norm, max_grad_norm)
            self.x["word"] = torch.clamp(
                self.x["word"] + grad_word,
                -self.clamp_value, self.clamp_value
            )
            self.x["pos"] = torch.clamp(
                self.x["pos"] + grad_pos,
                -self.clamp_value, self.clamp_value
            )
            weight_update_word = torch.zeros_like(self.layers["word"].weight)
            weight_update_pos = torch.zeros_like(self.layers["pos"].weight)
            for b in range(error.size(0)):
                for s in range(error.size(1)):
                    idx_w = self.input_ids[b, s]
                    idx_p = self.position_ids[b, s]
                    weight_update_word[idx_w] += self.local_lr * error[b, s]
                    weight_update_pos[idx_p] += self.local_lr * error[b, s]
            weight_update_word = torch.clamp(weight_update_word, -max_grad_norm, max_grad_norm)
            weight_update_pos = torch.clamp(weight_update_pos, -max_grad_norm, max_grad_norm)
            self.layers["word"].weight.data.add_(weight_update_word)
            self.layers["pos"].weight.data.add_(weight_update_pos)
            self.layers["word"].weight.data.clamp_(-self.weight_clamp, self.weight_clamp)
            self.layers["pos"].weight.data.clamp_(-self.weight_clamp, self.weight_clamp)
        elif self.kind in ("fc1", "fc2", "output_attn", "final_output"):
            grad_x = self.local_lr * (error @ self.layers["main"].weight)
            grad_x = torch.clamp(grad_x, -max_grad_norm, max_grad_norm)
            self.x = torch.clamp(
                self.x + grad_x,
                -self.clamp_value, self.clamp_value
            )
            weight_update = self.local_lr * torch.einsum("bsh,bsv->hv", error, self.x)
            weight_update = torch.clamp(weight_update, -max_grad_norm, max_grad_norm)
            self.layers["main"].weight.data.add_(weight_update)
            self.layers["main"].weight.data.clamp_(-self.weight_clamp, self.weight_clamp)
            if self.update_bias and self.layers["main"].bias is not None:
                bias_update = self.local_lr * error.mean(dim=(0, 1))
                bias_update = torch.clamp(bias_update, -max_grad_norm, max_grad_norm)
                self.layers["main"].bias.data.add_(bias_update)
                self.layers["main"].bias.data.clamp_(-self.weight_clamp, self.weight_clamp)
        elif self.kind == "attn":
            grad_x = -self.local_lr * error
            grad_x = torch.clamp(grad_x, -max_grad_norm, max_grad_norm)
            self.x = torch.clamp(
                self.x + grad_x,
                -self.clamp_value, self.clamp_value
            )
            for key in ["q", "k", "v"]:
                proj_in = self.x.detach()
                weight_update = self.local_lr * torch.einsum("bsh,bsv->hv", error, proj_in)
                weight_update = torch.clamp(weight_update, -max_grad_norm, max_grad_norm)
                self.layers[key].weight.data.add_(weight_update)
                self.layers[key].weight.data.clamp_(-self.weight_clamp, self.weight_clamp)
                if self.update_bias and self.layers[key].bias is not None:
                    bias_update = self.local_lr * error.mean(dim=(0, 1))
                    bias_update = torch.clamp(bias_update, -max_grad_norm, max_grad_norm)
                    self.layers[key].bias.data.add_(bias_update)
                    self.layers[key].bias.data.clamp_(-self.weight_clamp, self.weight_clamp)

        if self.is_holding_error and self.energy_fn:
            energy = self.energy_fn(mu, target_activity).mean().item()
            if np.isnan(energy) or np.isinf(energy):
                print(f"Warning: NaN or inf in energy for {self.kind}")
            else:
                self._energy = energy
        self._errors.append({"step": len(self._errors), "type": self.kind, "error": error.mean().item()})

        if self.kind == "embed":
            self._x_cache["word"] = self.x["word"].detach()
            self._x_cache["pos"] = self.x["pos"].detach()
        else:
            self._x_cache[self.kind] = self.x.detach()
            if self.kind in ("fc1", "fc2", "output_attn", "final_output"):
                self._W_cache[self.kind] = self.layers["main"].weight.data.clone()
            elif self.kind == "attn":
                for key in ["q", "k", "v"]:
                    self._W_cache[key] = self.layers[key].weight.data.clone()

    def get_x(self, kind: str):
        return self.x if kind == self.kind else None

    def get_energy(self):
        return self._energy

    def clear_energy(self):
        self._energy = None

    def clear_errors(self):
        self._errors = []
