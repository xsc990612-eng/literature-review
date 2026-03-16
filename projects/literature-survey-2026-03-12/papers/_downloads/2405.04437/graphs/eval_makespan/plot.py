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

def get_yticks_seconds(model, pdratio):
    if "yi-6b" in model.lower():
        return np.arange(0, 2500, 600) if pdratio == 500 else np.arange(0, 3300, 800) if pdratio == 100 else np.arange(0, 4900, 1200)
    elif "llama-3-8b" in model.lower():
        return np.arange(0, 1700, 400) if pdratio == 500 else np.arange(0, 2100, 400) if pdratio == 100 else np.arange(0, 2600, 500)
    elif "yi-34b" in model.lower():
        return np.arange(0, 4900, 1200) if pdratio == 500 else np.arange(0, 7700, 1900) if pdratio == 100 else np.arange(0, 13000, 3000)
    else:
        return np.arange(0, 2500, 600)

def get_yticks_minutes(model, pdratio):
    if "yi-6b" in model.lower():
        return np.arange(0, 2500, 600) if pdratio == 500 else np.arange(0, 3300, 800) if pdratio == 100 else np.arange(0, 4900, 1200)
    elif "llama-3-8b" in model.lower():
        return np.arange(0, 31, 5) if pdratio == 500 else np.arange(0, 37, 6) if pdratio == 100 else np.arange(0, 49, 8)
    elif "yi-34b" in model.lower():
        return np.arange(0, 4900, 1200) if pdratio == 500 else np.arange(0, 7700, 1900) if pdratio == 100 else np.arange(0, 13000, 3000)
    else:
        return np.arange(0, 2500, 600)

seq_lens = [32768, 65536, 131072]
seq_lens_ticks = ["32K", "64K", "128K"]

def plt_figure(df, name, pdratio, xlabel, ylabel, yticks, legend):
    fig, ax = plt.subplots(figsize=(16, 16))
    width = 0.20
    df = df[df["p_d"] == pdratio]
    df = df[df["cl"].isin(seq_lens)]
    x_pos = np.arange(len(seq_lens))
    fontsize=66
    df['FA_Paged'] = df['FA_Paged'] / 60
    df['FI_Paged'] = df['FI_Paged'] / 60
    df['FA_vAttention'] = df['FA_vAttention'] / 60
    ax.bar(x_pos - 1.5*width, df["FA_vAttention"], width, label="FA_vAttention", color=colors["FA_vAttention"], hatch=hatches["FA_vAttention"], alpha=opacity, edgecolor='black')
    #perf = abs(df["FA_vAttention"])
    #for i, v in enumerate(perf):
    #    ax.text(i - 1.5*width, v, f"{1.0:.2f}x", color='black', ha='center', va='bottom', fontsize=36, rotation=90)
    
    ax.bar(x_pos - 0.5*width, df["FA_Paged"], width, label="FA_Paged", color=colors["FA_Paged"], hatch=hatches["FA_Paged"], alpha=opacity, edgecolor='black')
    perf, rel_perf = abs(df["FA_Paged"]), df["FA_Paged"] / df["FA_vAttention"]
    for i, v in enumerate(perf):
        ax.text(i - 0.5*width, v, f"{rel_perf.iloc[i]:.2f}x", color='black', ha='center', va='bottom', fontsize=fontsize-8, rotation=90)
    
    ax.bar(x_pos + 0.5*width, df["FI_Paged"], width, label="FI_Paged", color=colors["FI_Paged"], hatch=hatches["FI_Paged"], alpha=opacity, edgecolor='black')
    perf, rel_perf = abs(df["FI_Paged"]), df["FI_Paged"] / df["FA_vAttention"]
    for i, v in enumerate(perf):
        ax.text(i + 0.5*width, v, f"{rel_perf.iloc[i]:.2f}x", color='black', ha='center', va='bottom', fontsize=fontsize-8, rotation=90)

    plt.xticks(x_pos, seq_lens_ticks, fontsize=fontsize)
    #plt.yticks(np.arange(0, ymax, ygap), fontsize=36)
    plt.yticks(yticks, fontsize=fontsize)
    ax.grid(axis='y', linestyle='--')
    if xlabel:
        ax.set_xlabel('Context Length', fontweight='bold', fontsize=fontsize)
    if ylabel:
        plt.ylabel("Makespan (minutes)", fontweight='bold', fontsize=fontsize)
    if legend:
        plt.legend(loc='upper left', ncol=1, frameon=True, fontsize=fontsize)
    #plt.title(f"{name} (P:D={pdratio})", fontweight='bold', fontsize=36)
    plt.title(f"P:D={pdratio}", fontweight='bold', fontsize=fontsize+8)
    plt.tight_layout()
    plt.savefig(f"{name}_makespan_{pdratio}_pagesize.pdf")

def plot_runtime_overhead():
    #models = ["yi-6B.csv", "llama-3-8B.csv", "yi-34B.csv"]
    models = ["llama-3-8B.csv"]
    for model in models:
        df = pd.read_csv(model, sep="\t")
        #print(df)
        #continue
        name = get_model_name(model)
        for pdratio in [500, 100, 50]:
            yticks = get_yticks_minutes(model, pdratio)
            ylabel = True if pdratio == 500 else False
            xlabel = True if model == models[-1] else False
            legend = True if model == models[0] and pdratio == 500 else False
            plt_figure(df, name, pdratio, xlabel, ylabel, yticks, legend)

plot_runtime_overhead()