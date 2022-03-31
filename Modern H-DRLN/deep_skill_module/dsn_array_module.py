import torch.nn as nn


class DSNArrayModule(nn.Module):
    def __init__(self, DSN_array):
        super(DSNArrayModule, self).__init__()

        self.dsn_array = nn.ModuleList(DSN_array)

    def forward(self, x):
        return [nn(x) for nn in self.dsn_array]

    def get_save_dict(self):
        return {"dsn_array": self.dsn_array}
