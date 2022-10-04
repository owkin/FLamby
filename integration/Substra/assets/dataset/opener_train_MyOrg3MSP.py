import substratools as tools


class FlambyTrainOpener(tools.Opener):
    def get_X(self, folders):
        config = {"center": 2, "train": True}
        return config

    def get_y(self, folders):
        pass

    def fake_X(self, n_samples=None):
        pass

    def fake_y(self, n_samples=None):
        pass
