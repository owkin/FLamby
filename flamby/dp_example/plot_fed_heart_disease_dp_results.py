# Plot
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

sns.set_theme(style="darkgrid")


results = pd.read_csv("results_fed_heart_disease.csv")

linestyle_tuple = [
    ("loosely dotted", (0, (1, 10))),
    ("dotted", (0, (1, 1))),
    ("densely dotted", (0, (1, 1))),
    ("loosely dashed", (0, (5, 10))),
    ("dashed", (0, (5, 5))),
    ("densely dashed", (0, (5, 1))),
    ("loosely dashdotted", (0, (3, 10, 1, 10))),
    ("dashdotted", (0, (3, 5, 1, 5))),
    ("densely dashdotted", (0, (3, 1, 1, 1))),
    ("dashdotdotted", (0, (3, 5, 1, 5, 1, 5))),
    ("loosely dashdotdotted", (0, (3, 10, 1, 10, 1, 10))),
    ("densely dashdotdotted", (0, (3, 1, 1, 1, 1, 1))),
]

deltas = [d for d in results["d"].unique() if d is not None]
for i, d in enumerate(deltas):
    cdf = results.loc[results["d"] == d]
    fig = sns.lineplot(
        data=cdf, x="e", y="perf", label=f"delta={d}", linestyle=linestyle_tuple[i][1]
    )
    fig.set(xscale="log")
fig.axhline(
    np.array(results.loc[results[d].isnull(), "d"].tolist()).mean(), color="black"
)
plt.legend()
plt.xlabel("epsilon")
plt.ylabel("Perf")
plt.savefig("trial.png")
