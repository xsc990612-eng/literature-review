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

context_lengths = ["1K", "2K", "4K", "8K", "16K", "32K"]

models = ['01-ai/Yi-6B-200K', '01-ai/Yi-34B-200K']
batch_sizes = [1, 2, 4, 8]
context_length = 16384

def model_to_title(model):
    if "yi-6b" in model.lower():
        return "Yi-6B (1 A100)"
    elif "yi-34b" in model.lower():
        return "Yi-34B (2 A100s)"
    elif "llama" in model.lower():
        return "LLAMA-70B (4 H100s)"
    else:
        return model

def config_to_color(config):
    if config == 'vLLM':
        return colors[3]
    if config == 'FLASH':
        return colors[1]
    if config == 'FLASH_PAGED':
        return colors[0]
    if config == 'FLASH_vATTN_v2':
        return colors[1]

def get_config_to_hatch(config):
    if config == 'vLLM':
        return hatches[3]
    if config == 'FLASH':
        return hatches[3]
    if config == 'FLASH_PAGED':
        return hatches[1]
    if config == 'FLASH_vATTN_v2':
        return hatches[3]

def model_to_y_ticks(model):
    if "yi-6b" in model.lower():
        return 520, 100
    if "yi-34b" in model.lower():
        return 520, 100
    if "llama" in model.lower():
        return 160, 25
    else:
        return 500, 50

configs = ['vLLM', 'FLASH_PAGED', 'FLASH']
legend = ['vLLM', 'FA-Paged', 'FA'] #, 'vLLM-FA(Paged)', 'vAttention-FA']

def plot_common(df, model1, model2, model3):
    fig, axs = plt.subplots(ncols=3, figsize=(30, 8))
    width = 0.20

    models = [model1, model2, model3]
    for ax_index, model in enumerate(models):
        ax = axs[ax_index]
        df_model = df[df['model'] == model]
        x_pos = np.arange(len(batch_sizes))
        for i, bs in enumerate(batch_sizes):
            df_bs = df_model[df_model['bs'] == bs]
            ax.bar(i - 3*width/2, df_bs['vLLM'], width, color=config_to_color('vLLM'), alpha=opacity, hatch=get_config_to_hatch('vLLM'), edgecolor='black')
            bars = ax.bar(i-width/2, df_bs['FLASH_PAGED'], width, color=config_to_color('FLASH_PAGED'), alpha=opacity, hatch=get_config_to_hatch('FLASH_PAGED'), edgecolor='black')
            speedup = f"{df_bs['FLASH_PAGED'].values[0] / df_bs['vLLM'].values[0]:.2f}"
            ax.text(i - width/2, bars[0].get_height(), speedup + 'x', ha='center', va='bottom', fontsize=28, rotation=90)
            bars2 = ax.bar(i + width/2, df_bs['FLASH'], width, color=config_to_color('FLASH'), alpha=opacity, hatch=get_config_to_hatch('FLASH'), edgecolor='black')
            speedup = f"{df_bs['FLASH'].values[0] / df_bs['vLLM'].values[0]:.2f}"
            ax.text(i + width/2, bars2[0].get_height(), speedup + 'x', ha='center', va='bottom', fontsize=28, rotation=90)

        x_ticks = ['1*16K', '2*16K', '4*16K', '8*16K']
        ax.set_xticks(x_pos)
        ax.set_xticklabels(x_ticks, fontsize=24)
        y_max, y_ticks = model_to_y_ticks(model)
        ax.set_yticks(np.arange(0, y_max, y_ticks))
        ax.tick_params(labelsize=24)
        ax.grid(axis='y', linestyle='--')
        ax.set_xlabel('Batch Size * Context Length', fontsize=28)
        ax.set_ylabel("Kernel Latency (us)", fontweight='bold', fontsize=32)
        ax.set_title(model_to_title(model), fontweight='bold')
        if ax_index == 0:
            ax.legend(legend, loc='upper left', fontsize='32', ncols=1)

    plt.tight_layout()
    plt.savefig(f"combined_gqa_latency.pdf")

df = pd.read_csv("data.csv", sep=";")
plot_common(df, 'yi-6B', 'yi-34B', 'llama2-70B')