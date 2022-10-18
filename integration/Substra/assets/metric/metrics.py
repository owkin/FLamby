import numpy as np
import substratools as tools

from flamby.datasets.fed_tcga_brca import metric


class TCGABRCAMetrics(tools.Metrics):
    def score(self, inputs, outputs):
        y_true = inputs["y"]
        y_pred = self.load_predictions(inputs["predictions"])

        perf = float(metric(y_true, y_pred))
        tools.save_performance(perf, outputs["performance"])

    def load_predictions(self, path):
        return np.load(path)


if __name__ == "__main__":
    tools.metrics.execute(TCGABRCAMetrics())
