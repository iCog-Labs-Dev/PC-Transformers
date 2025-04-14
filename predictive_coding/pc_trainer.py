import torch.nn as nn
from .config import PCTrainerConfig
from .optimizer_utils import recreate_optimizer_p, recreate_optimizer_x
from .pc_utils import get_model_xs, model_has_pc_layers, preprocess_step_index_list

class PCTrainer:
    def __init__(self, config: PCTrainerConfig):
        self.config = config
        self._model = config.model
        self._T = config.T
        self._optimizer_x = None
        self._optimizer_p = None

        self._update_x_at = preprocess_step_index_list(config.update_x_at, self._T)
        self._update_p_at = preprocess_step_index_list(config.update_p_at, self._T)

        self.recreate_optimizers()

    def recreate_optimizers(self):
        if model_has_pc_layers(self._model):
            self._optimizer_x = recreate_optimizer_x(
                get_model_xs(self._model),
                self.config.optimizer_x_fn,
                self.config.optimizer_x_kwargs,
                self.config.manual_optimizer_x_fn
            )
        else:
            self._optimizer_x = recreate_optimizer_x(
                self._model.parameters(),
                self.config.optimizer_x_fn,
                self.config.optimizer_x_kwargs,
                self.config.manual_optimizer_x_fn
            )

        self._optimizer_p = recreate_optimizer_p(
            self._model.parameters(),
            self.config.optimizer_p_fn,
            self.config.optimizer_p_kwargs,
            self.config.manual_optimizer_p_fn
        )

    @property
    def model(self):
        return self._model

    @property
    def T(self):
        return self._T

    @property
    def optimizer_x(self):
        return self._optimizer_x

    @property
    def optimizer_p(self):
        return self._optimizer_p