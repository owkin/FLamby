import substratools as tools


class FlambyTrainOpener(tools.Opener):
    def get_data(self, folders):
        config = {"center": 5, "train": True}
        return config

    def fake_data(self, n_samples=None):
        pass
