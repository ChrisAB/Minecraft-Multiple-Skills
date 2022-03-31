import torch
import torch.nn as nn
import torch.nn.functional as F
from neural_networks.q_network import QNetwork
from control_blocks.basic_control_block import ControlBlock


class DSNModule(nn.Module):
    def __init__(self, DSN_module, q_network=QNetwork(), control_block=ControlBlock()):
        super(DSNModule, self).__init__()

        self.dsn = DSN_module
        self.q_network = q_network
        self.control_block = control_block

    def forward(self, x):
        nn_output = self.dsn(x)
        best_skill_decider = self.q_network(x)

        return self.control_block.output(best_skill_decider, nn_output)
