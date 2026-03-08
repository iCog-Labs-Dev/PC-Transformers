import torch
import torch.nn as nn
import torch.nn.functional as F


class HopfieldMemory(nn.Module):
    """
    Continuous Hopfield-style associative memory used for memory-based
    initialization in predictive coding layers.

    Retrieval:
        h0 = softmax(delta * ((o Q) + b) K^T) V

    Updates are local/manual (no autograd optimizer step).
    """

    def __init__(
        self,
        obs_dim: int,
        hidden_dim: int,
        memory_slots: int = 128,
        delta: float = 8.0,
    ):
        super().__init__()
        self.obs_dim = int(obs_dim)
        self.hidden_dim = int(hidden_dim)
        self.memory_slots = int(memory_slots)

        self.Q = nn.Parameter(torch.empty(self.obs_dim, self.hidden_dim))
        self.K = nn.Parameter(torch.empty(self.memory_slots, self.hidden_dim))
        self.V = nn.Parameter(torch.empty(self.memory_slots, self.hidden_dim))
        self.b = nn.Parameter(torch.zeros(self.hidden_dim))
        self.delta = nn.Parameter(torch.tensor(float(delta)))

        nn.init.xavier_uniform_(self.Q)
        nn.init.xavier_uniform_(self.K)
        nn.init.xavier_uniform_(self.V)

        self._bootstrapped = False

    def memory_loss(self, observation: torch.Tensor, converged_state: torch.Tensor) -> torch.Tensor:
        """Squared-error memory loss LH = ||h0 - hT||^2."""
        retrieved, _, _, _ = self.retrieve(observation)
        return ((retrieved - converged_state) ** 2).mean()

    def _project_query(self, observation: torch.Tensor) -> torch.Tensor:
        return observation @ self.Q + self.b

    def retrieve(self, observation: torch.Tensor):
        """
        observation: [B, S, obs_dim]
        returns:
            h0: [B, S, hidden_dim]
            attn: [B, S, memory_slots]
            query: [B, S, hidden_dim]
            logits: [B, S, memory_slots]
        """
        query = self._project_query(observation)
        logits = torch.matmul(query, self.K.t())
        attn = F.softmax(self.delta.clamp(min=0.1, max=100.0) * logits, dim=-1)
        h0 = torch.matmul(attn, self.V)
        return h0, attn, query, logits

    def bootstrap_value_matrix(self, observation: torch.Tensor, ifw_state: torch.Tensor) -> None:
        """
        One-time pseudo-inverse initialization for V:
            V = A^+ h_ifw
        where A = softmax(delta * ((oQ + b)K^T)).
        """
        if self._bootstrapped:
            return

        with torch.no_grad():
            _, attn, _, _ = self.retrieve(observation)
            A = attn.reshape(-1, self.memory_slots)
            H = ifw_state.reshape(-1, self.hidden_dim)
            A_pinv = torch.linalg.pinv(A)
            V_new = A_pinv @ H
            if V_new.shape == self.V.shape:
                self.V.data.copy_(V_new)
            self._bootstrapped = True

    def local_update(
        self,
        observation: torch.Tensor,
        converged_state: torch.Tensor,
        lr: float,
        clamp_value: float,
        update_qk: bool = True,
    ) -> float:
        """
        Local manual update (no autograd step) to reduce ||h0 - hT||^2.
        """
        with torch.no_grad():
            retrieved, attn, query, logits = self.retrieve(observation)
            err = converged_state - retrieved
            lh = float((err ** 2).mean().item())

            n = float(max(observation.shape[0] * observation.shape[1], 1))
            flat_attn = attn.reshape(-1, self.memory_slots)
            flat_err = err.reshape(-1, self.hidden_dim)

            # Value update aligns stored patterns with converged hidden states.
            dV = flat_attn.t() @ flat_err / n
            self.V.data.add_(torch.clamp(lr * dV, -clamp_value, clamp_value))

            if update_qk:
                # Approximate local credit assignment for query/key projections.
                err_query = flat_err @ self.V.data.t() @ self.K.data
                flat_obs = observation.reshape(-1, self.obs_dim)
                dQ = flat_obs.t() @ err_query / n
                dK = flat_attn.t() @ err_query / n
                db = err_query.mean(dim=0)

                self.Q.data.add_(torch.clamp(lr * dQ, -clamp_value, clamp_value))
                self.K.data.add_(torch.clamp(lr * dK, -clamp_value, clamp_value))
                self.b.data.add_(torch.clamp(lr * db, -clamp_value, clamp_value))

                flat_logits = logits.reshape(-1, self.memory_slots)
                score = (flat_logits * flat_attn).mean()
                self.delta.data.add_(torch.clamp(lr * 0.01 * score, -0.1, 0.1))
                self.delta.data.clamp_(0.1, 100.0)

            return lh
