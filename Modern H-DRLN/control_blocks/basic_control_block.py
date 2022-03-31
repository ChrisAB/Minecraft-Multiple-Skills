import torch


class ControlBlock:
    def __init__(self):
        pass

    def output(self, q_network_output, dsn_output):
        _, index = torch.max(q_network_output)
        return dsn_output[index]

        # max = 0, max_index = 0
        # for i in enumerate(decision_block):
        #     if(decision_block[i] > max):
        #         max = decision_block[i]
        #         max_index = i

        # return dsn_output[max_index]
