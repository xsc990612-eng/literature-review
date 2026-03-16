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

#batch_sizes = [1, 2, 4, 8, 16, 32, 64, 128]
batch_sizes = [1]
#seq_lens = [1024, 2048, 4096, 8192, 16384]
prefill_seq_lens = ["2K", "4K", "8K", "16K", "32K"]
decode_seq_lens = ["1*32K", "2*32K", "4*32K", "8*32K", "16*32K"]

def attn_to_title(attn_type):
    if "mha" in attn_type:
        return "MHA (LLaMA2-7B)"
    if "gqa" in attn_type:
        return "GQA (Yi-34B)"
    if "mqa" in attn_type:
        return "MQA (StartCoder-15.5B)"

def plt_figure(df, phase):
    fig, ax = plt.subplots(figsize=(12, 8))
    width = 0.30
    x_pos = np.arange(len(prefill_seq_lens)) if phase == "prefill" else np.arange(len(decode_seq_lens))
    ax.bar(x_pos - 0.5*width, df["2MB"], width, label="2MB", color=colors["2MB"], hatch=hatches["2MB"], alpha=opacity, edgecolor='black')
    bars1 = ax.bar(x_pos + 0.5*width, df["64KB"], width, label="64KB", color=colors["64KB"], hatch=hatches["64KB"], alpha=opacity, edgecolor='black')
    perf = df["64KB"].values
    for i, v in enumerate(perf):
        ax.text(i + 0.5*width, v + 0.05, f"{v:.2f}x", color='black', ha='center', va='bottom', fontsize=32, rotation=90)

    if phase == "prefill":
        plt.xticks(x_pos, prefill_seq_lens, fontsize=32)
    else:
        plt.xticks(x_pos, decode_seq_lens, fontsize=32)
    plt.yticks(np.arange(0, 1.9, 0.20), fontsize=32)
    ax.grid(axis='y', linestyle='--')
    x_label = 'Context Length (Prefill)' if phase == "prefill" else 'Batch Size * Context Length (Decode)'
    ax.set_xlabel(x_label, fontweight='bold', fontsize=36)
    plt.ylabel("Normalized Runtime", fontweight='bold', fontsize=36)
    plt.legend(loc='best', ncol=2, frameon=True, fontsize=36)
    plt.tight_layout()
    plt.savefig(f"eval_micro_{phase}_pagesize.pdf")

def plot_runtime_overhead():
    df = pd.read_csv("prefill.csv", sep=";")
    plt_figure(df[df["model"] == "llama-3-8B"], "prefill")
    df = pd.read_csv("decode.csv", sep=";")
    plt_figure(df[df["model"] == "llama-3-8B"], "decode")

plot_runtime_overhead()