# Plot
import matplotlib.pyplot as plt

# import numpy as np
import pandas as pd
import seaborn as sns

sns.set_theme(style="darkgrid")
# datasets_names = ["Fed-Heart-Disease", "Fed-Camelyon16", "Fed-ISIC2019"]
# n_repetitions = 5
# rows = []
# for d in datasets_names:
#     for se in np.arange(42, 42 + n_repetitions):
#         rows.append({"dataset": d, "seed": se, "perf": float(np.random.uniform(0., 1., 1)), "finetune": True})
#         rows.append({"dataset": d, "seed": se, "perf": float(np.random.uniform(0., 1., 1)), "finetune": False})

# results = pd.DataFrame.from_dict(rows)


results = pd.read_csv("results_perso_vs_normal.csv")
results = results.rename(columns={"perf": "Perf"})
fig, ax = plt.subplots()
sns.boxplot(
    data=results,
    x="dataset",
    y="Perf",
    hue="finetune",
    ax=ax,
)
ax.set_xlabel(None)
ax.set_ylim(0.0, 1.0)

plt.legend()

plt.savefig("perso_vs_non_perso.pdf", dpi=100, bbox_inches="tight")
