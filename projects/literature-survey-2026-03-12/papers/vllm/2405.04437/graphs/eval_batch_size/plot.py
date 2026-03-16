import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import sys

plt.rcParams.update({'font.size': 40})
plt.rcParams.update({'font.family': 'Sans Serif'})
hatches = ['\\\\', '--', '//', '', 'xx', '\\\\']
colors = ['chocolate', 'cadetblue', 'wheat', 'gray', 'lightgreen']
opacity=0.65

models = ['Yi-6B', 'Llama-3-8B', 'Yi-34B']

colors = {
    '2MB': 'lightgreen',
    '256KB': 'cadetblue',
    '128KB': 'chocolate',
    '64KB': 'wheat',
}

hatches = {
    '2MB': '',
    '256KB': '\\',
    '128KB': '-',
    '64KB': '',
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

legend = ['2MB', '256KB', '128KB', '64KB']

def plot_common(df):
    fix, ax = plt.subplots(figsize=(18, 8))
    width = 0.12
    x_pos = np.arange(len(models))
    for i, model in enumerate(models):
        df_model = df[df['model'] == model]
        bars1 = ax.bar(i - 1.5*width, df_model['2MB'], width, color=colors['2MB'], alpha=opacity, hatch=hatches['2MB'], edgecolor='black')
        bars2 = ax.bar(i - 0.5*width, df_model['256KB'], width, color=colors['256KB'], alpha=opacity, hatch=hatches['256KB'], edgecolor='black')
        ratio = f"{df_model['256KB'].values[0] / df_model['2MB'].values[0]:.2f}"
        ax.text(i - 0.5*width, bars2[0].get_height(), ratio + 'x', ha='center', va='bottom', fontsize=28, rotation=90)
        bars3 = ax.bar(i + 0.5*width, df_model['128KB'], width, color=colors['128KB'], alpha=opacity, hatch=hatches['128KB'], edgecolor='black')
        ratio = f"{df_model['128KB'].values[0] / df_model['2MB'].values[0]:.2f}"
        ax.text(i + 0.5*width, bars3[0].get_height(), ratio + 'x', ha='center', va='bottom', fontsize=28, rotation=90)
        bars4 = ax.bar(i + 1.5*width, df_model['64KB'], width, color=colors['64KB'], alpha=opacity, hatch=hatches['64KB'], edgecolor='black')
        ratio = f"{df_model['64KB'].values[0] / df_model['2MB'].values[0]:.2f}"
        ax.text(i + 1.5*width, bars4[0].get_height(), ratio + 'x', ha='center', va='bottom', fontsize=28, rotation=90)

    ax.set_xticks(x_pos)
    ax.set_xticklabels(models)
    y_max, y_ticks = 410, 100
    ax.set_yticks(np.arange(0, y_max, y_ticks))
    #ax.tick_params(labelsize=32)
    ax.grid(axis='y', linestyle='--')
    ax.set_ylabel("Max Batch Size", fontweight='bold', fontsize=36)
    lg = ax.legend(legend, loc='upper right', fontsize=34, ncols=4)
    plt.tight_layout()
    plt.savefig(f"page_size_batch_size.pdf")

df = pd.read_csv("data.csv", sep=";")
plot_common(df)
