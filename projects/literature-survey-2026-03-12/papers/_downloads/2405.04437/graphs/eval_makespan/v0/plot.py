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

def get_yticks(model, pdratio):
    if "yi-6b" in model.lower():
        return np.arange(0, 2500, 600) if pdratio == 500 else np.arange(0, 3300, 800) if pdratio == 100 else np.arange(0, 4900, 1200)
    elif "llama-3-8b" in model.lower():
        return np.arange(0, 1700, 400) if pdratio == 500 else np.arange(0, 3300, 800) if pdratio == 100 else np.arange(0, 4900, 1200)
    elif "yi-34b" in model.lower():
        return np.arange(0, 4900, 1200) if pdratio == 500 else np.arange(0, 7700, 1900) if pdratio == 100 else np.arange(0, 13000, 3000)
    else:
        return np.arange(0, 2500, 600)

seq_lens = [32768, 65536, 131072]
seq_lens_ticks = ["32K", "64K", "128K"]

def plt_figure(df, name, pdratio, xlabel, ylabel, yticks, legend):
    fig, ax = plt.subplots(figsize=(16, 8))
    width = 0.20
    df = df[df["p_d"] == pdratio]
    df = df[df["cl"].isin(seq_lens)]
    x_pos = np.arange(len(seq_lens))

    ax.bar(x_pos - 1.5*width, df["FA_vAttention"], width, label="FA_vAttention", color=colors["FA_vAttention"], hatch=hatches["FA_vAttention"], alpha=opacity, edgecolor='black')
    #perf = abs(df["FA_vAttention"])
    #for i, v in enumerate(perf):
    #    ax.text(i - 1.5*width, v, f"{1.0:.2f}x", color='black', ha='center', va='bottom', fontsize=36, rotation=90)
    
    ax.bar(x_pos - 0.5*width, df["FA_Paged"], width, label="FA_Paged", color=colors["FA_Paged"], hatch=hatches["FA_Paged"], alpha=opacity, edgecolor='black')
    perf, rel_perf = abs(df["FA_Paged"]), df["FA_Paged"] / df["FA_vAttention"]
    for i, v in enumerate(perf):
        ax.text(i - 0.5*width, v, f"{rel_perf.iloc[i]:.2f}x", color='black', ha='center', va='bottom', fontsize=36, rotation=90)
    
    ax.bar(x_pos + 0.5*width, df["FI_Paged"], width, label="FI_Paged", color=colors["FI_Paged"], hatch=hatches["FI_Paged"], alpha=opacity, edgecolor='black')
    perf, rel_perf = abs(df["FI_Paged"]), df["FI_Paged"] / df["FA_vAttention"]
    for i, v in enumerate(perf):
        ax.text(i + 0.5*width, v, f"{rel_perf.iloc[i]:.2f}x", color='black', ha='center', va='bottom', fontsize=36, rotation=90)

    plt.xticks(x_pos, seq_lens_ticks, fontsize=36)
    #plt.yticks(np.arange(0, ymax, ygap), fontsize=36)
    plt.yticks(yticks, fontsize=36)
    ax.grid(axis='y', linestyle='--')
    if xlabel:
        ax.set_xlabel('Context Length', fontweight='bold', fontsize=36)
    if ylabel:
        plt.ylabel("Makespan (seconds)", fontweight='bold', fontsize=36)
    if legend:
        plt.legend(loc='upper left', ncol=1, frameon=True, fontsize=40)
    plt.title(f"{name} (P:D={pdratio})", fontweight='bold', fontsize=36)
    plt.tight_layout()
    plt.savefig(f"{name}_makespan_{pdratio}_pagesize.pdf")

def plot_runtime_overhead():
    models = ["yi-6B.csv", "llama-3-8B.csv", "yi-34B.csv"]
    for model in models:
        df = pd.read_csv(model, sep="\t")
        #print(df)
        #continue
        name = get_model_name(model)
        for pdratio in [500, 100, 50]:
            yticks = get_yticks(model, pdratio)
            ylabel = True if pdratio == 500 else False
            xlabel = True if "yi-34b" in model.lower() else False
            legend = True if "yi-6b" in model.lower() and pdratio == 500 else False
            plt_figure(df, name, pdratio, xlabel, ylabel, yticks, legend)

plot_runtime_overhead()