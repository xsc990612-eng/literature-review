import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import sys

plt.rcParams.update({'font.size': 28})
plt.rcParams.update({'font.family': 'Sans Serif'})
#hatches = ['\\\\', '--', '//', '', 'xx', '\\\\']
#colors = ['chocolate', 'cadetblue', 'wheat', 'gray', 'lightgreen']
opacity=0.65

opacity=0.65

colors = {
    '2MB': 'lightgreen',
    #'64KB': 'cadetblue',
    '64KB': 'chocolate',
    'fi': 'chocolate',
    'fi_paged': 'wheat',
    '32': 'gray'
}

hatches = {
    '2MB': '',
    #'64KB': '\\\\',
    '64KB': '-',
    'fi': '-',
    'fi_paged': '//',
    '32': ''
}

batch_sizes = [1, 2, 4, 8, 16, 32, 64]
#batch_sizes = [1]
seq_len = 32768
decode_seq_lens = ["1*32K", "2*32K", "4*32K", "8*32K", "16*2K", "32*32K", "64*32K"]

def attn_to_title(attn_type):
    if "mha" in attn_type:
        return "MHA (LLaMA2-7B)"
    if "gqa" in attn_type:
        return "GQA (Yi-34B)"
    if "mqa" in attn_type:
        return "MQA (StartCoder-15.5B)"

def plt_figure(df):
    fig, ax = plt.subplots(figsize=(21, 10))
    width = 0.30
    x_pos = np.arange(len(decode_seq_lens))
    df = df[df["context_length"] == seq_len]
    ax.bar(x_pos - 0.5*width, df["2MB_norm"], width, label="2MB", color=colors["2MB"], hatch=hatches["2MB"], alpha=opacity, edgecolor='black')
    bars1 = ax.bar(x_pos + 0.5*width, df["64KB_norm"], width, label="64KB", color=colors["64KB"], hatch=hatches["64KB"], alpha=opacity, edgecolor='black')
    perf = df["64KB_norm"].values
    for i, v in enumerate(perf):
        ax.text(i + 0.5*width, v + 0.05, f"{v:.2f}x", color='black', ha='center', va='bottom', fontsize=32, rotation=90)

    plt.xticks(x_pos, decode_seq_lens, fontsize=32)
    plt.yticks(np.arange(0, 1.9, 0.20), fontsize=32)
    ax.grid(axis='y', linestyle='--')
    x_label = 'Batch Size * Context Length (Decode)'
    ax.set_xlabel(x_label, fontweight='bold', fontsize=36)
    plt.ylabel("Normalized Runtime", fontweight='bold', fontsize=36)
    plt.legend(loc='best', ncol=2, frameon=True, fontsize=36)
    plt.tight_layout()
    plt.savefig(f"eval_pagesize_gpt3.pdf")

def plot_runtime_overhead():
    df = pd.read_csv("data.csv", sep="\t")
    df["2MB_norm"] = 1.0
    df["64KB_norm"] = df["64KB"] / df["2MB"]
    plt_figure(df[df["model"] == "gpt-3-175B"])

plot_runtime_overhead()