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

colors = {
    '1': 'gray',
    '2': 'cadetblue',
    '4': 'chocolate',
    '8': 'wheat',
    '16': 'lightgreen'
}

hatches = {
    '1': '',
    '2': '\\\\',
    '4': '-',
    '8': '//',
    '16': ''
}

seq_labels = ["2K", "4K", "8K", "16K", "32K"]
seq_lens = [2048, 4096, 8192, 16384, 32768]
batch_sizes = [1, 2, 4, 8, 16]

def plt_figure(df, model):
    fig, ax = plt.subplots(figsize=(15, 6))
    width = 0.15
    df = df[df['cl'].isin(seq_lens)]
    x_pos = np.arange(len(seq_lens))
    for i, bs in enumerate(batch_sizes):
        ax.bar(x_pos + (i-2)*width, df[df['bs'] == bs]['overhead'], width, label=bs, color=colors[str(bs)], alpha=opacity, hatch=hatches[str(bs)], edgecolor='black')
    
    plt.xticks(x_pos, seq_labels, fontsize=36)
    plt.yticks(np.arange(0, 22, 4), fontsize=36)
    ax.grid(axis='y', linestyle='--')
    ax.set_xlabel('Context Length', fontweight='bold', fontsize=26)
    plt.ylabel("% Overhead", fontweight='bold', fontsize=32)
    plt.legend(loc='upper center', ncols=5, frameon=True, fontsize=32)
    plt.tight_layout()
    plt.savefig(f"{model}_prefill_overhead.pdf")

df = pd.read_csv("data.csv", sep=";")
plt_figure(df, "flashinfer")
