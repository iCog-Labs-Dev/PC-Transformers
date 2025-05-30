import torch.nn as nn
from predictive_coding.pc_layer import PCLayer
class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.fc1 = nn.Linear(config.n_embed, 4 * config.n_embed)
        self.fc2 = nn.Linear(4 * config.n_embed, config.n_embed)
        # self.dropout = nn.Dropout(config.dropout)

        self.pc_layer2 = PCLayer(
            T=config.T,
            local_learning_rate=config.local_learning_rate,
            is_holding_error=config.is_holding_error,
            update_bias = config.update_bias,
            energy_fn_name=config.energy_fn_name,
        )

        self.pc_layer1 = PCLayer(
            T=config.T,
            local_learning_rate=config.local_learning_rate,
            is_holding_error=config.is_holding_error,
            update_bias = config.update_bias,
            energy_fn_name=config.energy_fn_name,
            
        )
        
    def forward(self, output_x):
        self.pc_layer2(target_activity = output_x, layer = self.fc2, layer_type = "fc2")
        fc2_x = self.pc_layer2.get_x("fc2")

        self.pc_layer1(target_activity = fc2_x, layer = self.fc1, layer_type = "fc1")
        fc1_x = self.pc_layer1.get_x("fc1")

        return fc1_x
    
    def evaluate(self, x):
         fc_1 = self.fc1(x)
         fc_2 = self.fc2(fc_1)

         return fc_2
