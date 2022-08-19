# Plot
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

sns.set_theme(style="darkgrid")


results = pd.read_csv("results_fed_heart_disease.csv")
results = results.rename(columns={"perf": "Performance"})
linestyle_str = [
    ("solid", "solid"),  # Same as (0, ()) or '-'
    ("dotted", "dotted"),  # Same as (0, (1, 1)) or ':'
    ("dashed", "dashed"),  # Same as '--'
    ("dashdot", "dashdot"),
]
linestyle_tuple = [
    ("loosely dotted", (0, (1, 10))),
    ("densely dotted", (0, (1, 1))),
    ("loosely dashed", (0, (5, 10))),
    ("densely dashed", (0, (5, 1))),
    ("loosely dashdotted", (0, (3, 10, 1, 10))),
    ("densely dashdotted", (0, (3, 1, 1, 1))),
    ("dashdotdotted", (0, (3, 5, 1, 5, 1, 5))),
    ("loosely dashdotdotted", (0, (3, 10, 1, 10, 1, 10))),
    ("densely dashdotdotted", (0, (3, 1, 1, 1, 1, 1))),
]
linestyles = linestyle_tuple + linestyle_str
deltas = [d for d in results["d"].unique() if not (np.isnan(d))]
fig, ax = plt.subplots()
for i, d in enumerate(deltas):
    cdf = results.loc[results["d"] == d]
    sns.lineplot(
        data=cdf,
        x="e",
        y="Performance",
        label=f"delta={d}",
        linestyle=linestyles[::-1][i][1],
        ax=ax,
    )
ax.set_xscale("log")
xtick_values = [d for d in results["e"].unique() if not (np.isnan(d))]
xlabels = [str(v) for v in xtick_values]
ax.set_xticks(xtick_values, xlabels)
ax.axhline(
    np.array(results.loc[results["d"].isnull(), "Performance"].tolist()).mean(),
    color="black",
    label="Baseline wo DP",
)
ax.set_xlim(0.1, 50)
plt.legend()
plt.xlabel("epsilon")
plt.ylabel("Performance")
plt.savefig("perf_function_of_dp_heart_disease.pdf", dpi=100, bbox_inches="tight")
