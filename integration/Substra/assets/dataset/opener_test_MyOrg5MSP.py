import substratools as tools
from torch.utils.data import DataLoader

from flamby.datasets.fed_tcga_brca import FedTcgaBrca


class FlambyTestOpener(tools.Opener):
    def get_data(self, folders):
        config = {"center": 4, "train": False}
        return config

    def fake_data(self, n_samples=None):
        pass
