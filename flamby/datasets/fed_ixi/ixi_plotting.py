from flamby.datasets.fed_ixi import IXITinyRaw
from tqdm import tqdm
import nibabel as nib
import numpy as np

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy import stats

raw = IXITinyRaw()

def plot_histogram(axis, array, num_positions=100, label=None, alpha=0.05, color=None):
    values = array.ravel()
    kernel = stats.gaussian_kde(values)
    positions = np.linspace(values.min(), values.max(), num=num_positions)
    histogram = kernel(positions)
    kwargs = dict(linewidth=1, color='black' if color is None else color, alpha=alpha)
    if label is not None:
        kwargs['label'] = label
    axis.plot(positions, histogram, **kwargs)

plt.style.use('seaborn')
fig, ax = plt.subplots(dpi=300, figsize=(20, 10))
for path in tqdm(raw.images_paths):
    array = np.array(nib.load(path).dataobj)
    if 'HH' in path.name: color = 'red'
    elif 'Guys' in path.name: color = 'green'
    elif 'IOP' in path.name: color = 'blue'
    plot_histogram(ax, array, color=color)
ax.set_xlim(-100, 2000)
ax.set_ylim(0, 0.004);
ax.set_xlabel('Intensity', fontsize=24)
ax.set_ylabel('Density', fontsize=24)
ax.tick_params(axis='x', labelsize=19)
ax.tick_params(axis='y', labelsize=19)
legend_elements = [Line2D([0], [0], color='r', lw=4, label='HH'),
                   Line2D([0], [0], color='g', lw=4, label='Guys'),
                   Line2D([0], [0], color='b', lw=4, label='IOP')]
ax.legend(handles=legend_elements, loc='upper right',fontsize=22)
ax.grid()
plt.savefig('histograms_ixi.pdf', bbox_inches='tight')