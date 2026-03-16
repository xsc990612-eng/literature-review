import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import sys

plt.rcParams.update({'font.size': 24})
plt.rcParams.update({'font.family': 'Sans Serif'})
hatches = ['\\\\', '--', '//', '', 'xx', '\\\\']
colors = ['chocolate', 'cadetblue', 'wheat', 'gray', 'lightgreen']
opacity=0.75

context_lengths = ["1K", "2K", "4K", "8K", "16K", "32K"]
configs = ["Prefill", "2MB", "4MB", "8MB", "16MB", "CudaMalloc", "TorchCachingAlloc"]


def model_to_title(model):
    if model == "llama":
        return "LLaMA2-7B"
    if model == "yi":
        return "Yi-34B"
    if model == "sc":
        return "SC-15.5B"

def plt_figure(df, model):
    fig, ax = plt.subplots(figsize=(15, 9))
    seq_lens = df['CL'].unique()
    for cfg in ['2MB', '4MB', '8MB', '16MB']:
        df[cfg] = df[cfg] + df['Prefill']
        df['overhead_' + cfg] = df[cfg] / df['Prefill']
    width = 0.15
    x_pos = np.arange(len(seq_lens))
    ax.bar(np.arange(len(seq_lens)) - 3*width, df['Prefill'], width, label='PyTorch', color=colors[3], alpha=opacity-0.2, hatch=hatches[3], edgecolor='black')
    ax.bar(np.arange(len(seq_lens)) - 2*width, df['2MB'], width, label='2MB', color=colors[1], alpha=opacity, hatch=hatches[5], edgecolor='black')
    ax.bar(np.arange(len(seq_lens)) - 1*width, df['4MB'], width, label='4MB', color=colors[0], alpha=opacity-0.2, hatch=hatches[1], edgecolor='black')
    ax.bar(np.arange(len(seq_lens)) - 0*width, df['8MB'], width, label='8MB', color=colors[2], alpha=opacity, hatch=hatches[2], edgecolor='black')
    ax.bar(np.arange(len(seq_lens)) + 1*width, df['16MB'], width, label='16MB', color=colors[4], alpha=opacity-0.2, hatch=hatches[3], edgecolor='black')
    #for i, v in enumerate(df['Prefill']):
    #    ax.text(i - 3*width, v + 0.01, '1', color='black', fontweight='bold', ha='center', va='bottom', fontsize=20, rotation=90)
    for cfg, shift in zip(['2MB', '4MB', '8MB', '16MB'], range(-2, 2)):
        for i, v in enumerate(df[cfg]):
            ax.text(i + shift*width*1.05, v, str(df['overhead_'+cfg].iloc[i]), color='black', ha='center', va='bottom', fontsize=24, rotation=90)

    plt.xticks(x_pos, seq_lens, fontsize=36)
    #plt.yticks(np.arange(0, 1000, 100), fontsize=28)
    ax.grid(axis='y', linestyle='--')
    #plt.yticks(10.0**np.arange(-2, 4), fontsize=28)
    #ax.set_yscale('log')
    ax.set_xlabel('Context Length', fontweight='bold', fontsize=32)
    plt.ylabel("Time (ms)", fontweight='bold', fontsize=32)
    plt.legend(loc='best', ncols=2, frameon=True, fontsize=32)
    plt.title(model_to_title(model), fontweight='bold')
    plt.tight_layout()
    plt.savefig(f"{model}_prefill_overhead.pdf")

for model in ["llama", "yi", "sc"]:
    df = pd.read_csv(f"{model}.csv", sep=";")
    plt_figure(df, model)

#plot_runtime_overhead()