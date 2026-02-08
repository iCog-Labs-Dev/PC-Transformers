import functools
from typing import Dict, Any, Optional
import torch

from .sgd import sgd_step, sgd_init
from .adam import adam_step, adam_init


def get_opt_init_fn(opt="adam"):
    return {
        "adam": adam_init,
        "sgd": sgd_init,
    }[opt]


def get_opt_step_fn(opt="adam", **kwargs):
    return {
        "adam": functools.partial(adam_step, **kwargs),
        "sgd": functools.partial(sgd_step, **kwargs),
    }[opt]


class PCOptimizer:
    """Lightweight optimizer for predictive coding weight updates (Adam/SGD)."""

    def __init__(
        self,
        opt_name: str = "adam",
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps: float = 1e-8,
    ) -> None:
        self.opt_name = opt_name.lower()
        self.beta1 = float(beta1)
        self.beta2 = float(beta2)
        self.eps = float(eps)
        self._state: Dict[int, Any] = {}

    def _init_state(self, param: torch.Tensor):
        init_fn = get_opt_init_fn(self.opt_name)
        return init_fn([param])

    def _sync_state_device(self, state, param: torch.Tensor):
        if self.opt_name == "adam":
            g1, g2, time_step = state
            if g1[0].device != param.device:
                g1 = [g.to(param.device) for g in g1]
            if g2[0].device != param.device:
                g2 = [g.to(param.device) for g in g2]
            if time_step.device != param.device:
                time_step = time_step.to(param.device)
            return g1, g2, time_step
        if state.device != param.device:
            return state.to(param.device)
        return state

    def step_param(
        self,
        param: torch.Tensor,
        update: torch.Tensor,
        lr: float,
        clamp_value: Optional[float] = None,
    ) -> None:
        """Apply a single optimizer step to a parameter tensor."""
        if update is None:
            return

        update = update.to(param.device, dtype=param.dtype)
        lr = float(lr)

        state = self._state.get(id(param))
        if state is None:
            state = self._init_state(param)
        state = self._sync_state_device(state, param)

        if self.opt_name == "adam":
            step_fn = get_opt_step_fn(
                self.opt_name,
                eta=lr,
                beta1=self.beta1,
                beta2=self.beta2,
                eps=self.eps,
            )
        else:
            step_fn = get_opt_step_fn(self.opt_name, eta=lr)

        with torch.no_grad():
            new_state, new_params = step_fn(state, [param], [update])
            new_param = new_params[0]
            delta = new_param - param
            if clamp_value is not None:
                delta = torch.clamp(delta, -abs(clamp_value), abs(clamp_value))
                new_param = param + delta
            param.data.copy_(new_param)

        self._state[id(param)] = new_state
