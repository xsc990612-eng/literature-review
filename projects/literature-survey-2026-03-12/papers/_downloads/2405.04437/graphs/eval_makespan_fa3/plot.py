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
    'FA2_Paged': 'chocolate',
    'FA2_vAttention': 'lightgreen',
    'FA3_vAttention': 'wheat',
}

hatches = {
    'FA2_Paged': '-',
    'FA2_vAttention': '\\',
    'FA3_vAttention': '',
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

num_requests = 427
def plot_throughput(df):
    fig, ax = plt.subplots(figsize=(16, 7))
    models = df["model"]
    width = 0.20
    x_pos = np.arange(len(models))
    fontsize = 36
    df["FA2_Paged_throughput"] = (num_requests) / df["FA2_Paged"]
    df["FA2_vAttention_throughput"] = (num_requests) / df["FA2_vAttention"]
    df["FA3_vAttention_throughput"] = (num_requests) / df["FA3_vAttention"]
    ax.bar(x_pos - 1.5*width, df["FA2_Paged_throughput"], width, label="FA2_Paged", color=colors["FA2_Paged"], hatch=hatches["FA2_Paged"], alpha=opacity, edgecolor='black')
    for i, v in enumerate(df["FA2_Paged_throughput"]):
        ax.text(i - 1.5*width, v + 0.1, f"{v:.2f}", color='black', ha='center', va='bottom', fontsize=fontsize-10, rotation=0)  # Add absolute value
    ax.bar(x_pos - 0.5*width, df["FA2_vAttention_throughput"], width, label="FA2_vAttention", color=colors["FA2_vAttention"], hatch=hatches["FA2_vAttention"], alpha=opacity, edgecolor='black')
    for i, v in enumerate(df["FA2_vAttention_throughput"]):
        ax.text(i - 0.5*width, v + 0.1, f"{v:.2f}", color='black', ha='center', va='bottom', fontsize=fontsize-10, rotation=0)  # Add absolute value
    ax.bar(x_pos + 0.5*width, df["FA3_vAttention_throughput"], width, label="FA3_vAttention", color=colors["FA3_vAttention"], hatch=hatches["FA3_vAttention"], alpha=opacity, edgecolor='black')
    for i, v in enumerate(df["FA3_vAttention_throughput"]):
        ax.text(i + 0.5*width, v + 0.1, f"{v:.2f}", color='black', ha='center', va='bottom', fontsize=fontsize-10, rotation=0)  # Add absolute value
    plt.xticks(x_pos, models_ticks, fontsize=fontsize)
    plt.yticks(np.arange(0, 13, 2), fontsize=fontsize-4)
    ax.grid(axis='y', linestyle='--')
    plt.ylabel("Requests per minute", fontweight='bold', fontsize=fontsize-10)
    plt.legend(loc='upper right', ncol=1, frameon=True, fontsize=fontsize-4)
    plt.tight_layout()
    plt.savefig(f"throughput_arxiv_h100.pdf")

def plot_throughput_horizontal(df):
    fig, ax = plt.subplots(figsize=(16, 7))
    models = df["model"]
    width = 0.20
    y_pos = np.arange(len(models))
    fontsize = 36
    df["FA2_Paged_throughput"] = (num_requests) / df["FA2_Paged"]
    df["FA2_vAttention_throughput"] = (num_requests) / df["FA2_vAttention"]
    df["FA3_vAttention_throughput"] = (num_requests) / df["FA3_vAttention"]
    ax.barh(y_pos - 1.5*width, df["FA2_Paged_throughput"], width, label="FA2_Paged", color=colors["FA2_Paged"], hatch=hatches["FA2_Paged"], alpha=opacity, edgecolor='black')
    for i, v in enumerate(df["FA2_Paged_throughput"]):
        ax.text(v + 0.1, i - 1.5*width, f"{v:.2f}", color='black', va='center', ha='left', fontsize=fontsize-10, rotation=0)  # Add absolute value
    ax.barh(y_pos - 0.5*width, df["FA2_vAttention_throughput"], width, label="FA2_vAttention", color=colors["FA2_vAttention"], hatch=hatches["FA2_vAttention"], alpha=opacity, edgecolor='black')
    for i, v in enumerate(df["FA2_vAttention_throughput"]):
        ax.text(v + 0.1, i - 0.5*width, f"{v:.2f}", color='black', va='center', ha='left', fontsize=fontsize-10, rotation=0)  # Add absolute value
    ax.barh(y_pos + 0.5*width, df["FA3_vAttention_throughput"], width, label="FA3_vAttention", color=colors["FA3_vAttention"], hatch=hatches["FA3_vAttention"], alpha=opacity, edgecolor='black')
    for i, v in enumerate(df["FA3_vAttention_throughput"]):
        ax.text(v + 0.1, i + 0.5*width, f"{v:.2f}", color='black', va='center', ha='left', fontsize=fontsize-10, rotation=0)  # Add absolute value
    plt.yticks(y_pos, models_ticks, fontsize=fontsize)
    plt.xticks(np.arange(0, 13, 2), fontsize=fontsize-4)
    ax.grid(axis='x', linestyle='--')
    plt.xlabel("Requests per minute", fontweight='bold', fontsize=fontsize-10)
    plt.legend(loc='upper right', ncol=1, frameon=True, fontsize=fontsize-4)
    plt.tight_layout()
    plt.savefig(f"throughput_arxiv_h100.pdf")

df = pd.read_csv("data.csv", sep=";")
plot_throughput(df)