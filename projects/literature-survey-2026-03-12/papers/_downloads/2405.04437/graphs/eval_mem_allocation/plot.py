import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import sys

plt.rcParams.update({'font.size': 24})
plt.rcParams.update({'font.family': 'Sans Serif'})
hatches = ['\\\\', '--', '//', '', 'xx', '\\\\']
colors = ['chocolate', 'cadetblue', 'wheat', 'gray', 'lightgreen']
opacity=0.65

context_lengths = ["1K", "2K", "4K", "8K", "16K", "32K"]

block_sizes = ['64KB', '128KB', '256KB', '2MB']

colors = {
    '64KB': 'gray',
    '128KB': 'chocolate',
    '256KB': 'cadetblue',
    '2MB': 'wheat',
}

hatches = {
    '64KB': '',
    '128KB': '\\\\',
    '256KB': '--',
    '2MB': '//',
}

def model_to_y_ticks(model):
    if "yi-6b" in model.lower():
        return 520, 100
    if "yi-34b" in model.lower():
        return 520, 100
    if "llama" in model.lower():
        return 160, 25
    else:
        return 500, 50

legend = ['64KB', '128KB', '256KB', '2MB'] 

def plot_common(df):
    fix, ax = plt.subplots(figsize=(15, 6))
    width = 0.15
    configs = ['TP-1', 'TP-2']
    x_pos = np.arange(len(configs))
    for i, cfg in enumerate(configs):
        df_cfg = df[df['config'] == cfg]
        bars1 = ax.bar(i - 1.5*width, df_cfg['64KB'], width, color=colors['64KB'], alpha=opacity, hatch=hatches['64KB'], edgecolor='black')
        bars2 = ax.bar(i - 0.5*width, df_cfg['128KB'], width, color=colors['128KB'], alpha=opacity, hatch=hatches['128KB'], edgecolor='black')
        bars1 = ax.bar(i + 0.5*width, df_cfg['256KB'], width, color=colors['256KB'], alpha=opacity, hatch=hatches['256KB'], edgecolor='black')
        bars2 = ax.bar(i + 1.5*width, df_cfg['2MB'], width, color=colors['2MB'], alpha=opacity, hatch=hatches['2MB'], edgecolor='black')

    ax.set_xticks(x_pos)
    ax.set_xticklabels(configs)
    y_max, y_ticks = 82, 10
    ax.set_yticks(np.arange(0, y_max, y_ticks))
    ax.tick_params(labelsize=32)
    ax.grid(axis='y', linestyle='--')
    #ax.set_xlabel('Batch Size * Context Length', fontsize=22, fontweight='bold')
    ax.set_ylabel("Memory Allocatation Rate \n (GB per second)", fontweight='bold', fontsize=26)
    #ax.set_title(model_to_title(model), fontweight='bold')
    lg = ax.legend(legend, loc='upper left', fontsize='26', ncols=4)

    plt.tight_layout()
    plt.savefig(f"vattn_mem_alloc_rate.pdf")

df = pd.read_csv("data.csv", sep=";")
plot_common(df)