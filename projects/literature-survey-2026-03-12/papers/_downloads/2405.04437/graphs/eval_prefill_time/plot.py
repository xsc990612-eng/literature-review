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
#models = ['Llama-3-8B', 'Yi-34B']

colors = {
    'vLLM': 'lightgreen',
    'vAttn-64KB': 'cadetblue',
    'vAttn-128KB': 'chocolate',
    'vAttn-256KB': 'cadetblue',
    'vAttn-2MB': 'chocolate',
    'vAttn': 'wheat'
}

hatches = {
    'vLLM': '',
    'vAttn-64KB': '\\',
    'vAttn-128KB': '-',
    'vAttn-256KB': '',
    'vAttn-2MB': '-',
    'vAttn': '/',
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

#legend = ['vLLM', '64KB-sync', '128KB-sync', '256KB-sync', '2MB-sync', 'vAttention']
legend = ['Without CUDA APIs', 'CUDA APIs + 64KB pages (synchronous) ', 'CUDA APIs + 2MB (synchronous)', 'CUDA APIs + Deferred reclamation']

def plot_common(df):
    fix, ax = plt.subplots(figsize=(18, 8))
    width = 0.12
    x_pos = np.arange(len(models))
    for i, model in enumerate(models):
        df_model = df[df['model'] == model]
        bars1 = ax.bar(i - 1.5*width, df_model['vLLM'], width, color=colors['vLLM'], alpha=opacity, hatch=hatches['vLLM'], edgecolor='black')
        bars2 = ax.bar(i - 0.5*width, df_model['vAttn-64KB'], width, color=colors['vAttn-64KB'], alpha=opacity, hatch=hatches['vAttn-64KB'], edgecolor='black')
        ratio = f"{df_model['vAttn-64KB'].values[0] / df_model['vLLM'].values[0]:.2f}"
        ax.text(i - 0.5*width, bars2[0].get_height(), ratio + 'x', ha='center', va='bottom', fontsize=28, rotation=90)
        #bars3 = ax.bar(i - 0.5*width, df_model['vAttn-128KB'], width, color=colors['vAttn-128KB'], alpha=opacity, hatch=hatches['vAttn-128KB'], edgecolor='black')
        #ratio = f"{df_model['vAttn-128KB'].values[0] / df_model['vLLM'].values[0]:.2f}"
        #ax.text(i - 0.5*width, bars3[0].get_height(), ratio + 'x', ha='center', va='bottom', fontsize=18, rotation=90)
        #bars4 = ax.bar(i + 0.5*width, df_model['vAttn-256KB'], width, color=colors['vAttn-256KB'], alpha=opacity, hatch=hatches['vAttn-256KB'], edgecolor='black')
        #ratio = f"{df_model['vAttn-256KB'].values[0] / df_model['vLLM'].values[0]:.2f}"
        #ax.text(i + 0.5*width, bars4[0].get_height(), ratio + 'x', ha='center', va='bottom', fontsize=18, rotation=90)
        bars5 = ax.bar(i + 0.5*width, df_model['vAttn-2MB'], width, color=colors['vAttn-2MB'], alpha=opacity, hatch=hatches['vAttn-2MB'], edgecolor='black')
        ratio = f"{df_model['vAttn-2MB'].values[0] / df_model['vLLM'].values[0]:.2f}"
        ax.text(i + 0.5*width, bars5[0].get_height(), ratio + 'x', ha='center', va='bottom', fontsize=28, rotation=90)
        bars6 = ax.bar(i + 1.5*width, df_model['vAttn'], width, color=colors['vAttn'], alpha=opacity, hatch=hatches['vAttn'], edgecolor='black')
        ratio = f"{df_model['vAttn'].values[0] / df_model['vLLM'].values[0]:.2f}"
        ax.text(i + 1.5*width, bars6[0].get_height(), ratio + 'x', ha='center', va='bottom', fontsize=28, rotation=90)

    ax.set_xticks(x_pos)
    ax.set_xticklabels(models)
    #y_max, y_ticks = 6100, 1000
    y_max, y_ticks = 7, 1
    ax.set_yticks(np.arange(0, y_max, y_ticks))
    #ax.tick_params(labelsize=32)
    ax.grid(axis='y', linestyle='--')
    ax.set_ylabel("Prefill Time (seconds)", fontweight='bold', fontsize=36)
    lg = ax.legend(legend, loc='upper left', fontsize=34, ncols=1)
    plt.tight_layout()
    plt.savefig(f"vattn_prefill_time.pdf")

df = pd.read_csv("data.csv", sep=";")
plot_common(df)
