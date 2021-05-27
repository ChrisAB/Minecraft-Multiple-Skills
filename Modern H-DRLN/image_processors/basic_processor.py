import numpy as np
import torchvision


class BasicImageProcessor:
    def __init__(self, args):
        self.args = args

    def transform(self, obs):
        return torchvision.transforms.functional.to_tensor(
            np.flip(obs, axis=0).copy()).reshape(1, 3, 64, 64)
