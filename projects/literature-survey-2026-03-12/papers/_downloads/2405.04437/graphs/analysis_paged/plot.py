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
    'fa': 'lightgreen',
    'fa_paged': 'cadetblue',
    'fi': 'chocolate',
    'fi_paged': 'wheat',
    '32': 'gray'
}

hatches = {
    'fa': '',
    'fa_paged': '\\\\',
    'fi': '-',
    'fi_paged': '//',
    '32': ''
}

#batch_sizes = [1, 2, 4, 8, 16, 32, 64, 128]
batch_sizes = [1]
#seq_lens = [1024, 2048, 4096, 8192, 16384]
seq_lens = ["1K", "2K", "4K", "8K", "16K", "32K"]


def attn_to_title(attn_type):
    if "mha" in attn_type:
        return "MHA (LLaMA2-7B)"
    if "gqa" in attn_type:
        return "GQA (Yi-34B)"
    if "mqa" in attn_type:
        return "MQA (StartCoder-15.5B)"

def plt_figure(df):
    fig, ax = plt.subplots(figsize=(15, 6))
    width = 0.15
    x_pos = np.arange(len(seq_lens))
    cl_configs = ["2K", "4K", "8K", "16K", "32K"]
    ax.bar(x_pos - 1.5*width, df["fa"], width, label="FA", color=colors["fa"], hatch=hatches["fa"], alpha=opacity, edgecolor='black')
    bars1 = ax.bar(x_pos - 0.5*width, df["fa_paged"], width, label="FA_Paged", color=colors["fa_paged"], hatch=hatches["fa_paged"], alpha=opacity, edgecolor='black')
    perf = df["fa_paged"] / df["fa"]
    for i, v in enumerate(perf):
        ax.text(i - 0.5*width, v + 0.05, f"{v:.2f}x", color='black', ha='center', va='bottom', fontsize=20, rotation=90)
    ax.bar(x_pos + 0.5*width, df["fi"], width, label="FI", color=colors["fi"], hatch=hatches["fi"], alpha=opacity, edgecolor='black')
    bars2 = ax.bar(x_pos + 1.5*width, df["fi_paged"], width, label="FI_Paged", color=colors["fi_paged"], hatch=hatches["fi_paged"], alpha=opacity, edgecolor='black')
    perf2 = df["fi_paged"] / df["fi"]
    positions = df["fi_paged"] / df["fa"]
    for i, v in enumerate(perf2):
        ax.text(i + 1.5*width, positions[i] + 0.05, f"{v:.2f}x", color='black', ha='center', va='bottom', fontsize=22, rotation=90)
    
    plt.xticks(x_pos, seq_lens, fontsize=28)
    plt.yticks(np.arange(0, 1.51, 0.25), fontsize=28)
    ax.grid(axis='y', linestyle='--')
    ax.set_xlabel('Context Length', fontweight='bold', fontsize=28)
    plt.ylabel("Normalized Runtime", fontweight='bold', fontsize=28)
    #plt.legend(loc='upper center', ncols=4, frameon=True, fontsize=26)
    #plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.55), ncol=4, frameon=True, fontsize=20)
    #plt.tight_layout()

    plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.35), ncol=4, frameon=True, fontsize=28)
    plt.tight_layout(rect=[0, 0, 1, 1.1])  # Adjust the rectangle's top to leave space for the legend

    plt.savefig("analysis_paged.pdf")

def plot_runtime_overhead():
    df = pd.read_csv("data.csv", sep=";")
    plt_figure(df[df["model"] == "llama-3-8B"])

plot_runtime_overhead()