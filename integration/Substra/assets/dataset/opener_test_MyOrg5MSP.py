import substratools as tools
from torch.utils.data import DataLoader

from flamby.datasets.fed_tcga_brca import FedTcgaBrca


class FlambyTestOpener(tools.Opener):
    def get_X(self, folders):
        config = self.get_config()
        return config

    def get_y(self, folders):
        config = self.get_config()
        dataset = FedTcgaBrca(**config)
        dataloader = DataLoader(dataset, batch_size=len(dataset))
        return next(iter(dataloader))[1]

    def get_config(self):
        return {"center": 4, "train": False}

    def fake_X(self, n_samples=None):
        pass

    def fake_y(self, n_samples=None):
        pass
