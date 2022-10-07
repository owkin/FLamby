import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from flamby.datasets.fed_heart_disease import FedHeartDisease
from flamby.utils import seaborn_styling

seaborn_styling(figsize=(15, 10), legend_fontsize=24, labelsize=20)
palette = sns.color_palette("colorblind", n_colors=16)

# get centers
X, y = {}, {}
for i in range(4):
    data = FedHeartDisease(center=i, train=True)
    X[i] = data.features
    y[i] = data.labels

    data = FedHeartDisease(center=i, train=False)
    X[i] += data.features
    y[i] += data.labels

    X[i] = np.array([[float(elt) for elt in x] for x in X[i]])
    y[i] = np.array([float(elt) for elt in y[i]])

full_X, full_y, center = [], [], []
for i in range(4):
    full_X += list(X[i])
    full_y += list(y[i])
    center += [i] * len(y[i])
full_X = np.array(full_X)
full_y = np.array(full_y)
center = np.array(center)

# plots
fig, ax = plt.subplots()
select = [0, 2, 5]  # 4]#np.arange(13)#[0, 2, 5]#, 7]
leg = ["age", "trestbps", "thalach", "oldpeak"]
hdls = []

for num, i in enumerate(select):
    for j in range(4):
        sns.kdeplot(X[j][:, i], color=palette[num], fill=palette[num], alpha=0.3)

ax.set_ylim(0, 0.06)
ax.set_xlim(20, 200)

for num, c in enumerate(select):
    (hdl,) = ax.plot([0, 0], [0, 0], color=palette[num])
    hdls += [hdl]
ax.legend(hdls, leg)
ax.set_xlabel("Value")

plt.savefig("heart_heterogeneity.pdf", bbox_inches="tight")
