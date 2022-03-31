import torch
import torch.nn as nn
import torch.nn.functional as F
from neural_networks.q_network import QNetwork
from control_blocks.basic_control_block import ControlBlock


class DSNSkillDistillationModule(nn.Module):
    def __init__(self, hidden_layers, outputs):
        super(DSNSkillDistillationModule, self).__init__()

        self.hidden_layers = hidden_layers
        self.outputs = nn.ModuleList(outputs)

    def forward(self, x):
        x = self.hidden_layers(x)
        return [output(x) for output in self.outputs]

    def get_save_dict(self):
        return {"hidden_layers": self.hidden_layers, "outputs": self.outputs}
