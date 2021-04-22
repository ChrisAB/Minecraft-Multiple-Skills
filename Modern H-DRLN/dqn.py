import torch
import torch.nn as nn
import torch.nn.functional as F


class DQN(nn.Module):
    def __init__(self, input_dimensions=[1*1, 84, 84], inputs=[32, 64, 64], filters=[8, 4, 3], strides=[4, 2, 1], number_of_hidden_neurons=[512], number_of_actions=16,  nl=nn.ReLU):
        super(DQN, self).__init__()
        self.number_of_hidden_neurons = number_of_hidden_neurons
        self.input_conv = nn.Conv2d(
            input_dimensions[0], inputs[0], filters[0], strides[0])
        self.input_nl = nl()
        self.conv_layers = nn.ModuleList()
        for i in range(1, len(inputs)):
            self.conv_layers.insert(2*(i-1),
                                    nn.Conv2d(inputs[i-1], inputs[i], filters[i], strides[i]))
            self.conv_layers.insert(2*(i-1)+1, nl())

        self.conv_layers = nn.Sequential(*self.conv_layers)
        nel = self.temp_forward(torch.zeros(1, *input_dimensions))

        self.input_linear = nn.Linear(
            torch.numel(nel), number_of_hidden_neurons[0])
        self.linear_nl = nl()
        if(len(number_of_hidden_neurons) > 1):
            self.linear_layers = nn.ModuleList()
            for i in range(1, len(number_of_hidden_neurons)):
                self.linear_layers.insert(
                    2*(i-1), nn.Linear(number_of_hidden_neurons[i-1], number_of_hidden_neurons[i]))
                self.linear_layers.insert(2*(i-1)+1, nl())
            self.linear_layers = nn.Sequential(*self.linear_layers)

        self.last_layer = nn.Linear(
            number_of_hidden_neurons[-1], number_of_actions)

    def temp_forward(self, x):
        res = self.input_conv(x)
        res = self.input_nl(res)
        res = self.conv_layers(res)
        return res

    def forward(self, x):
        res = self.input_conv(x)
        res = self.input_nl(res)
        res = self.conv_layers(res)
        res = res.reshape(res.size(0), 1024)
        res = self.input_linear(res)
        res = self.linear_nl(res)
        if(len(self.number_of_hidden_neurons) > 1):
            res = self.linear_layers(res)
        res = self.last_layer(res)
        return res

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


if __name__ == '__main__':
    inputs = [32, 64, 64]
    filters = [8, 4, 3]
    strides = [4, 2, 1]
    number_of_hidden_neurons = 512

    net = DQN()
