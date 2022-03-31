import torch


class SequentialQNetwork:
    def __init__(self, in_features=[3, 64, 64], number_DSN_outputs=10, number_actions=16):
        self.dense_1 = torch.nn.Linear(
            in_features=in_features, out_features=256)
        self.dense_2 = torch.nn.Linear(in_features=256, out_features=256)
        self.dense_3 = torch.nn.Linear(in_features=256, out_features=512)
        self.dense_4 = torch.nn.Linear(
            in_features=512, out_features=(number_DSN_outputs + number_actions))
        self.softmax = torch.nn.Softmax(dim=0)

    def forward(self, x):
        x = self.dense_1(x)
        x = self.dense_2(x)
        x = self.dense_3(x)
        x = self.dense_4(x)
        return self.softmax(x)
