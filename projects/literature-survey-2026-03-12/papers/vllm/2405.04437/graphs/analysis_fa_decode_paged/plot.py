import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import sys

plt.rcParams.update({'font.size': 24})
plt.rcParams.update({'font.family': 'Sans Serif'})
#hatches = ['\\\\', '--', '//', '', 'xx', '\\\\']
#colors = ['chocolate', 'cadetblue', 'wheat', 'gray', 'lightgreen']
opacity=0.65

opacity=0.65

colors = {
    '1': 'gray',
    '4': 'cadetblue',
    '8': 'chocolate',
    '16': 'wheat',
    '32': 'lightgreen'
}

hatches = {
    '1': '',
    '4': '\\\\',
    '8': '-',
    '16': '//',
    '32': ''
}

batch_sizes = [1, 2, 4, 8, 16, 32, 64, 128]
#seq_lens = [1024, 2048, 4096, 8192, 16384]
seq_lens = ["2K", "4K", "8K", "16K", "32K"]


def attn_to_title(attn_type):
    if "mha" in attn_type:
        return "MHA (LLaMA2-7B)"
    if "gqa" in attn_type:
        return "GQA (Yi-34B)"
    if "mqa" in attn_type:
        return "MQA (StartCoder-15.5B)"

def plt_figure(df, attn_type, key, ylabel, ymin, ymax, ygap, convert_to_percentage=False):
    df = df[df["bs"].isin(batch_sizes)]
    df = df[df["seqlen"].isin(seq_lens)]
    df[key] = np.maximum(df[key], 1)
    # some columns are in speedup format like 1.1, convert them to % first
    if convert_to_percentage:
        df[key] = df[key] - df[key].astype(int)
        df[key] = (df[key] * 100).astype(int)
    fig, ax = plt.subplots(figsize=(15, 6))
    bs_values = df['bs'].unique()
    width = 0.15
    x_pos = np.arange(len(seq_lens))
    bs_configs = [1, 4, 8, 16, 32]
    for idx, bs in enumerate(bs_configs):
        ax.bar(x_pos + (idx-2)*width, df[df['bs'] == bs][key], width, label=bs, color=colors[str(bs)], alpha=opacity, hatch=hatches[str(bs)], edgecolor='black')

    plt.xticks(x_pos, seq_lens, fontsize=36)
    plt.yticks(np.arange(ymin, ymax, ygap), fontsize=36)
    ax.grid(axis='y', linestyle='--')
    ax.set_xlabel('Context Length', fontweight='bold', fontsize=26)
    plt.ylabel(ylabel, fontweight='bold', fontsize=32)
    plt.legend(loc='best', ncols=5, frameon=True, fontsize=32)
    plt.tight_layout()
    plt.savefig("fa_decode_paged_" + attn_type + ".pdf")

def plot_runtime_overhead():
    attn_types = ["mha", "gqa", "mqa"]
    attn_types = ["gqa"]
    df = pd.read_csv("data.csv", sep=";")
    for attn_type in attn_types:
        plt_figure(df[df["attn_type"] == attn_type], attn_type, "decode_speedup", '% Overhead', 0, 18, 3, True)

plot_runtime_overhead()