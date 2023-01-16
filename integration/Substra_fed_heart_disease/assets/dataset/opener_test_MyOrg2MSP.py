import substratools as tools
from torch.utils.data import DataLoader

from flamby.datasets.fed_heart_disease import FedHeartDisease


class FlambyTestOpener(tools.Opener):
    def get_data(self, folders):
        config = {"center": 1, "train": False}
        return config

    def fake_data(self, n_samples=None):
        pass
