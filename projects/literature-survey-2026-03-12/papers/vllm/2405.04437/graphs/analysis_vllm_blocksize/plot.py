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

batch_sizes = [1, 2, 4, 8, 16]
context_length = 16384

def model_to_title(model):
    if "yi-6b" in model.lower():
        return "Yi-6B (1 A100)"
    elif "yi-34b" in model.lower():
        return "Yi-34B (2 A100s)"
    elif "llama" in model.lower():
        return "Llama-3-8B"
    else:
        return model

colors = {
    #'16': 'gray',
    '16': 'lightgreen',
    '32': 'cadetblue',
    '64': 'chocolate',
    '128': 'wheat'
}

hatches = {
    '16': '',
    '32': '\\\\',
    '64': '-',
    '128': '//'
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

legend = ['16', '32', '64', '128'] #, 'vLLM-FA(Paged)', 'vAttention-FA']

def plot_common(df):
    #fig, axs = plt.subplots(ncols=1, figsize=(30, 8))
    fix, ax = plt.subplots(figsize=(15, 5.5))
    width = 0.15

    x_pos = np.arange(len(batch_sizes))
    for i, bs in enumerate(batch_sizes):
        df_bs = df[df['bs'] == bs]
        bars1 = ax.bar(i - 1.5*width, df_bs['16'], width, color=colors['16'], alpha=opacity, hatch=hatches['16'], edgecolor='black')
        
        bars2 = ax.bar(i - 0.5*width, df_bs['32'], width, color=colors['32'], alpha=opacity, hatch=hatches['32'], edgecolor='black')
        speedup = f"{df_bs['32'].values[0] / df_bs['16'].values[0]:.2f}"
        ax.text(i - 0.5*width, bars2[0].get_height(), speedup + 'x', ha='center', va='bottom', fontsize=20, rotation=90)
        
        bars3 = ax.bar(i + 0.5*width, df_bs['64'], width, color=colors['64'], alpha=opacity, hatch=hatches['64'], edgecolor='black')
        speedup = f"{df_bs['64'].values[0] / df_bs['16'].values[0]:.2f}"
        ax.text(i + 0.5*width, bars3[0].get_height(), speedup + 'x', ha='center', va='bottom', fontsize=20, rotation=90)
        
        bars4 = ax.bar(i + 1.5*width, df_bs['128'], width, color=colors['128'], alpha=opacity, hatch=hatches['128'], edgecolor='black')
        speedup = f"{df_bs['128'].values[0] / df_bs['16'].values[0]:.2f}"
        ax.text(i + 1.5*width, bars4[0].get_height(), speedup + 'x', ha='center', va='bottom', fontsize=20, rotation=90)

    x_ticks = ['1*16K', '2*16K', '4*16K', '8*16K', '16*16K']
    plt.xticks(x_pos, x_ticks, fontsize=28)
    plt.yticks(np.arange(0, 3010, 500), fontsize=28)
    ax.grid(axis='y', linestyle='--')
    ax.set_xlabel('Batch Size * Context Length', fontsize=26, fontweight='bold')
    ax.set_ylabel("Kernel Latency (us)", fontweight='bold', fontsize=24)
    #ax.set_title(model_to_title(model), fontweight='bold')
    lg = ax.legend(legend, loc='upper left', fontsize='26', ncols=4, title='Block Size', title_fontsize='26')

    plt.tight_layout()
    plt.savefig(f"vllm_block_size.pdf")

df = pd.read_csv("data.csv", sep=";")
df = df[df['model'] == 'llama-3-8B']
plot_common(df)