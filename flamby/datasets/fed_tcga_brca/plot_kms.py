import itertools
import re

import matplotlib.pyplot as plt
import pandas as pd
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test
from torch.utils.data import DataLoader as dl

from flamby.datasets.fed_tcga_brca import NUM_CLIENTS, FedTcgaBrca
from flamby.utils import seaborn_styling

seaborn_styling(legend_fontsize=10, labelsize=12)

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
ax = kmf.plot()
ax.set_ylabel("Survival Probability")
plt.savefig("pooled_km.pdf", bbox_inches="tight")

plt.clf()
# Per center plot
kms = [
    KaplanMeierFitter().fit(y[:, 1], y[:, 0].astype("uint8"), label=f"Client {idx}")
    for idx, y in enumerate(local_ys)
]
axs = [km.plot() for km in kms]
axs[-1].set_ylabel("Survival Probability")
plt.savefig("local_kms.pdf", bbox_inches="tight")

# Adding logrank test table
columns = ["Local " + str(i) for i in range(NUM_CLIENTS)]
paired_labels = list(itertools.combinations(columns, 2))
print(paired_labels)
paired_times = list(itertools.combinations(local_ys, 2))

df_core = {col: [None] * len(columns) for col in columns}
df_core["paired_with"] = columns
df = pd.DataFrame(df_core)

for label_pair, times_pair in zip(paired_labels, paired_times):
    pval = logrank_test(
        times_pair[0][:, 1],
        times_pair[1][:, 1],
        event_observed_A=times_pair[0][:, 0],
        event_observed_B=times_pair[1][:, 0],
    ).p_value
    # We fill only the upper part of the symmetrical matrix
    if int(re.findall("(?<=Local )[0-9]{1}", label_pair[0])[0]) > int(
        re.findall("(?<=Local )[0-9]{1}", label_pair[1])[0]
    ):
        df.loc[(df["paired_with"] == label_pair[1]), label_pair[0]] = pval
    else:
        df.loc[(df["paired_with"] == label_pair[0]), label_pair[1]] = pval

df = df.set_index("paired_with")
df = df.drop(columns=["Local 0"])
df.drop(df.tail(1).index, inplace=True)

print(df.to_latex(index=True))
