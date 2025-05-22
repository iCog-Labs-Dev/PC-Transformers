import torch
import torch.nn  as nn
from predictive_coding.pc_layer import PCLayer



class OutputLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.output = nn.Linear(config.n_embed, config.vocab_size)
        self.pc_layer = PCLayer(
            T=config.T,
            local_learning_rate=config.local_learning_rate,
            energy_fn=config.energy_fn,
            x_init=config.x_init,
            is_holding_error=config.is_holding_error,
        )

    def forward(self, target_ids) -> torch.Tensor:
        self.pc_layer.clear_energy()
        self.pc_layer.clear_errors()

        output=self.pc_layer(target_activity = target_ids, layer=self.output, kind="final_output")
        output_x = self.pc_layer.get_x("final_output")

        return output_x