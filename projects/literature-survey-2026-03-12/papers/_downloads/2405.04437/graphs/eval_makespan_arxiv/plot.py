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
    'FI_Paged': 'chocolate',
    'FA_Paged': 'cadetblue',
    'FA_vAttention': 'lightgreen',
}

hatches = {
    'FI_Paged': '-',
    'FA_Paged': '\\',
    'FA_vAttention': '',
}

def get_model_name(model):
    if "yi-6b" in model.lower():
        return "Yi-6B"
    elif "llama-3-8b" in model.lower():
        return "Llama-3-8B"
    elif "yi-34b" in model.lower():
        return "Yi-34B"
    else:
        return model

models_ticks = ["Yi-6B", "Llama-3-8B", "Yi-34B"]

def plot_makespan(df):
    fig, ax = plt.subplots(figsize=(16, 7))
    models = df["model"]
    width = 0.20
    x_pos = np.arange(len(models))
    fontsize=36
    ax.bar(x_pos - 1.5*width, df["FA_vAttention"], width, label="FA2_vAttention", color=colors["FA_vAttention"], hatch=hatches["FA_vAttention"], alpha=opacity, edgecolor='black')
    ax.bar(x_pos - 0.5*width, df["FA_Paged"], width, label="FA2_Paged", color=colors["FA_Paged"], hatch=hatches["FA_Paged"], alpha=opacity, edgecolor='black')
    perf, rel_perf = abs(df["FA_Paged"]), df["FA_Paged"] / df["FA_vAttention"]
    for i, v in enumerate(perf):
        ax.text(i - 0.5*width, v, f"{rel_perf.iloc[i]:.2f}x", color='black', ha='center', va='bottom', fontsize=fontsize-8, rotation=90)
    ax.bar(x_pos + 0.5*width, df["FI_Paged"], width, label="FI_Paged", color=colors["FI_Paged"], hatch=hatches["FI_Paged"], alpha=opacity, edgecolor='black')
    perf, rel_perf = abs(df["FI_Paged"]), df["FI_Paged"] / df["FA_vAttention"]
    for i, v in enumerate(perf):
        ax.text(i + 0.5*width, v, f"{rel_perf.iloc[i]:.2f}x", color='black', ha='center', va='bottom', fontsize=fontsize-8, rotation=90)
    plt.xticks(x_pos, models_ticks, fontsize=fontsize)
    plt.yticks(np.arange(0, 430, 80), fontsize=fontsize-4)
    ax.grid(axis='y', linestyle='--')
    plt.ylabel("Makespan (minutes)", fontweight='bold', fontsize=fontsize-8)
    plt.legend(loc='upper left', ncol=2, frameon=True, fontsize=fontsize-4)
    plt.tight_layout()
    plt.savefig(f"makespan_arxiv.pdf")

num_requests = 427
def plot_throughput(df):
    fig, ax = plt.subplots(figsize=(16, 7))
    models = df["model"]
    width = 0.20
    x_pos = np.arange(len(models))
    fontsize = 36
    df["FA_vAttention_throughput"] = (num_requests) / df["FA_vAttention"]
    df["FA_Paged_throughput"] = (num_requests) / df["FA_Paged"]
    df["FI_Paged_throughput"] = (num_requests) / df["FI_Paged"]
    ax.bar(x_pos - 1.5*width, df["FA_Paged_throughput"], width, label="FA2_Paged", color=colors["FA_Paged"], hatch=hatches["FA_Paged"], alpha=opacity, edgecolor='black')
    for i, v in enumerate(df["FA_Paged_throughput"]):
        ax.text(i - 1.5*width, v + 0.1, f"{v:.2f}", color='black', ha='center', va='bottom', fontsize=fontsize-10, rotation=0)  # Add absolute value
    ax.bar(x_pos - 0.5*width, df["FI_Paged_throughput"], width, label="FI_Paged", color=colors["FI_Paged"], hatch=hatches["FI_Paged"], alpha=opacity, edgecolor='black')
    for i, v in enumerate(df["FI_Paged_throughput"]):
        ax.text(i - 0.5*width, v + 0.1, f"{v:.2f}", color='black', ha='center', va='bottom', fontsize=fontsize-10, rotation=0)  # Add absolute value
    ax.bar(x_pos + 0.5*width, df["FA_vAttention_throughput"], width, label="FA2_vAttention", color=colors["FA_vAttention"], hatch=hatches["FA_vAttention"], alpha=opacity, edgecolor='black')
    for i, v in enumerate(df["FA_vAttention_throughput"]):
        ax.text(i + 0.5*width, v + 0.1, f"{v:.2f}", color='black', ha='center', va='bottom', fontsize=fontsize-10, rotation=0)  # Add absolute value
    plt.xticks(x_pos, models_ticks, fontsize=fontsize)
    plt.yticks(np.arange(0, 7, 1), fontsize=fontsize-4)
    ax.grid(axis='y', linestyle='--')
    plt.ylabel("Requests per minute", fontweight='bold', fontsize=fontsize-10)
    plt.legend(loc='upper right', ncol=1, frameon=True, fontsize=fontsize-4)
    plt.tight_layout()
    plt.savefig(f"throughput_arxiv.pdf")

df = pd.read_csv("data.csv", sep=";")
plot_makespan(df)
plot_throughput(df)