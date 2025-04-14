import torch
import torch.nn as nn
from typing import Any, Callable, Dict, Optional, Union, List
from tqdm import trange

from .pc_utils import get_model_pc_layers, get_model_xs, compute_energies, check_model_training_state, model_has_pc_layers

def train_on_batch(
    trainer,
    inputs: Any,
    loss_fn: Optional[Callable] = None,
    loss_fn_kwargs: Dict = {},
    is_sample_x_at_batch_start: bool = True,
    is_reset_optimizer_x_at_batch_start: bool = True,
    is_reset_optimizer_p_at_batch_start: bool = False,
    is_unwrap_inputs: bool = False,
    is_optimize_inputs: bool = False,
    callback_before_forward: Optional[Callable] = None,
    callback_before_forward_kwargs: Dict = {},
    callback_after_backward: Optional[Callable] = None,
    callback_after_backward_kwargs: Dict = {},
    callback_after_t: Optional[Callable] = None,
    callback_after_t_kwargs: Dict = {},
    is_log_progress: bool = True,
    is_return_results_every_t: bool = True,
    is_checking_after_callback_after_t: bool = True,
    debug: Dict = {},
    backward_kwargs: Dict = {},
    is_clear_energy_after_use: bool = False,
    is_return_outputs: bool = False,
    energy_multipliers: Dict = {},
) -> Dict[str, List[Union[float, torch.Tensor]]]:
    
    """Train the model on a single batch of data."""
    
    model = trainer.model
    config = trainer.config
    model_pc_layers = list(get_model_pc_layers(model))
    model_xs = []
    results = {"loss": [], "energy": [], "overall": []}
    if is_return_outputs:
        results["outputs"] = []

    if check_model_training_state(model) is not True:
        raise RuntimeError("Model and all PCLayers must be in train mode before training.")

    is_dynamic_x_lr = (config.x_lr_discount < 1.0) or (config.x_lr_amplifier > 1.0)
    overalls = []
    unwrap_with = "**" if isinstance(inputs, dict) else "*" if isinstance(inputs, (tuple, list)) else ""
    t_iterator = trange(config.T) if is_log_progress else range(config.T)

    for t in t_iterator:
        if t == 0:
            if is_sample_x_at_batch_start:
                for pc_layer in model_pc_layers:
                    pc_layer.is_sample_x = True

            if is_optimize_inputs:
                inputs = nn.Parameter(inputs, requires_grad=True)

        if callback_before_forward:
            callback_before_forward(t, **callback_before_forward_kwargs)

        try:
            outputs = (model(**inputs) if unwrap_with == "**" else
                       model(*inputs) if unwrap_with == "*" else
                       model(inputs))
        except Exception as e:
            raise RuntimeError(f"Forward pass failed: {str(e)}")

        if t == 0:
            if is_sample_x_at_batch_start or is_reset_optimizer_x_at_batch_start:
                trainer.recreate_optimizers()

            if is_optimize_inputs:
                trainer.optimizer_x.param_groups[0]["params"].append(inputs)
            model_xs = list(get_model_xs(model))

            if is_reset_optimizer_p_at_batch_start:
                trainer.recreate_optimizers()

        if is_return_results_every_t or t == (config.T - 1):
            if is_return_outputs:
                results["outputs"].append(outputs.detach())

        # Loss computation
        loss = loss_fn(outputs, **loss_fn_kwargs) if loss_fn else None
        if loss and (is_return_results_every_t or t == config.T - 1):
            results["loss"].append(loss.item())

        # Energy computation
        energy = None
        if model_has_pc_layers(model):
            try:
                layer_energies = compute_energies(model, named=True)
                layer_energies = {
                    k: (v * energy_multipliers.get(k, 1.0))
                    for k, v in layer_energies.items() if not torch.isnan(v).any()
                }
                energy = sum(layer_energies.values())

                if is_clear_energy_after_use:
                    for pc_layer in model_pc_layers:
                        pc_layer.clear_energy()

                if is_return_results_every_t or t == (config.T - 1):
                    results["energy"].append(energy.item())
            except Exception as e:
                raise RuntimeError(f"Energy computation failed: {str(e)}")

        loss_x = sum(config.loss_x_fn(x) for x in model_xs).sum() if config.loss_x_fn else None
        loss_inputs = config.loss_inputs_fn(inputs) if config.loss_inputs_fn and is_optimize_inputs else None

        overall = torch.tensor(0.0, device=inputs.device)
        for part in [loss, energy * config.energy_coefficient if energy else None, loss_x, loss_inputs]:
            if part is not None:
                overall = overall + part

        if is_return_results_every_t or t == (config.T - 1):
            results["overall"].append(overall.item())

        early_stop = eval(config.early_stop_condition)
        if t in trainer._update_x_at and model_has_pc_layers(model):
            trainer.optimizer_x.zero_grad()

        if t in trainer._update_p_at or (early_stop and config.update_p_at_early_stop):
            trainer.optimizer_p.zero_grad()

        overall.backward(**backward_kwargs)
        if callback_after_backward:
            callback_after_backward(t, **callback_after_backward_kwargs)

        if t in trainer._update_x_at and model_has_pc_layers(model):
            trainer.optimizer_x.step()
            if is_dynamic_x_lr and len(overalls) >= 2:
                if overalls[-1] >= overalls[-2] and config.x_lr_discount < 1.0:
                    for pg in trainer.optimizer_x.param_groups:
                        pg["lr"] *= config.x_lr_discount
                elif config.x_lr_amplifier > 1.0:
                    for pg in trainer.optimizer_x.param_groups:
                        pg["lr"] *= config.x_lr_amplifier

        if t in trainer._update_p_at or (early_stop and config.update_p_at_early_stop):
            trainer.optimizer_p.step()

        if callback_after_t:
            callback_after_t(t, **callback_after_t_kwargs)
            if is_checking_after_callback_after_t and check_model_training_state(model) is not True:
                raise RuntimeError("Model was switched out of train mode during callback_after_t.")

        if is_log_progress:
            msg = "|"
            for name, val in zip(["l", "e", "x", "i", "o"], [loss, energy, loss_x, loss_inputs, overall]):
                if val is not None:
                    msg += f" {name}: {val:.3e} |"
            if is_dynamic_x_lr:
                msg += f" x_lrs: {[pg['lr'] for pg in trainer.optimizer_x.param_groups]} |"
            t_iterator.set_description(msg)

        if early_stop:
            break

    return results