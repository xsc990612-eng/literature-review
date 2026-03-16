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

models = ["Yi-6B", "Llama-3-8B", "Yi-34B"]
legend = ["Used", "Free"]

def model_to_title(model):
    if model == "llama":
        return "LLaMA2-7B"
    if model == "yi":
        return "Yi-34B"
    if model == "sc":
        return "SC-15.5B"

def plt_figure(df, figname):
    fig, ax = plt.subplots(figsize=(7, 5))
    all_models = df['model'].unique()
    width = 0.25
    x_pos = np.arange(len(models))
    for i, model in enumerate(all_models):
        df_model = df[df['model'] == model]
        ax.bar(i - 0.5 * width, df_model['used'], width, color=colors[0], alpha=opacity, hatch=hatches[3], edgecolor='black')
        ax.bar(i + 0.5 * width, df_model['free'], width, color=colors[1], alpha=opacity, hatch=hatches[3], edgecolor='black')

    plt.xticks(x_pos, all_models, fontsize=22)
    plt.yticks(np.arange(0, 155, 30), fontsize=22)
    ax.grid(axis='y', linestyle='--')
    #ax.set_xlabel('Model', fontweight='bold', fontsize=18)
    plt.ylabel("Memory (GB)", fontweight='bold', fontsize=22)
    plt.legend(legend, loc='upper left', ncols=1, frameon=True, fontsize=22)
    plt.tight_layout()
    plt.savefig(figname)

df = pd.read_csv("memutil.csv", sep=";")
plt_figure(df, "analysis_mem_utilization.pdf")
