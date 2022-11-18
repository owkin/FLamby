# Plot
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

sns.set_theme(style="darkgrid")

results = pd.read_csv("results_perso_vs_normal.csv")
results = results.rename(columns={"perf": "Performance"})
fig, ax = plt.subplots()
g = sns.boxplot(data=results, x="dataset", y="Performance", hue="finetune", ax=ax)
ax.set_xlabel(None)
ax.set_ylim(0.0, 1.0)
mapping_dict = {"True": "Fine-tuned", "False": "Not fine-tuned"}

handles, labels = ax.get_legend_handles_labels()

ax.legend(handles=handles, labels=[mapping_dict[lab] for lab in labels])


plt.savefig("perso_vs_non_perso.pdf", dpi=100, bbox_inches="tight")
