import matplotlib.pyplot as plt
import seaborn as sns
from lifelines import KaplanMeierFitter
from torch.utils.data import DataLoader as dl

from flamby.datasets.fed_tcga_brca import NUM_CLIENTS, FedTcgaBrca

sns.set()

pooled_training = FedTcgaBrca(train=True, pooled=True)
_, yp = [
    e.numpy()
    for e in iter(
        dl(pooled_training, batch_size=len(pooled_training), shuffle=False)
    ).next()
]
local_datasets = [
    FedTcgaBrca(center=i, pooled=False, train=True) for i in range(NUM_CLIENTS)
]
local_ys = [
    [
        e.numpy()
        for e in iter(
            dl(local_datasets[i], batch_size=len(local_datasets[i]), shuffle=False)
        ).next()
    ][1]
    for i in range(NUM_CLIENTS)
]


# create a kmf object
kmf = KaplanMeierFitter()

# Pooled plot
kmf.fit(yp[:, 1], yp[:, 0].astype("uint8"), label="KM Estimate for OS")
kmf.plot()
plt.savefig("pooled_km.png")

plt.clf()
# Per center plot
kms = [
    KaplanMeierFitter().fit(y[:, 1], y[:, 0].astype("uint8"), label=f"Local {idx}")
    for idx, y in enumerate(local_ys)
]
[km.plot() for km in kms]
plt.savefig("local_kms.png")
