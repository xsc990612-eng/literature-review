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
configs = ['vLLM_16', 'vLLM_128', 'FLASH_PAGED_16', 'FLASH_PAGED_128', 'FLASH_vATTN_v2']
legend = ['vLLM (bs=16)', 'vLLM (bs=128)', 'FA_Paged (bs=16)', 'FA_Paged (bs=128)', 'FA_vAttention']

models = ['Yi-6B-200K', 'Meta-Llama-3-8B', 'Yi-34B-200K']
context_length = 16384

colors = {
    'vLLM_16': 'chocolate',
    'vLLM_128': 'chocolate',
    'FLASH_PAGED_16': 'cadetblue',
    'FLASH_PAGED_128': 'cadetblue',
    'FLASH_vATTN_v2': 'wheat'
}

hatches = {
    'vLLM_16': '',
    'vLLM_128': '-',
    'FLASH_PAGED_16': '',
    'FLASH_PAGED_128': '//',
    'FLASH_vATTN_v2': ''
}

batchsizes = {
    'Yi-6B-200K': [1, 2, 4, 8, 16],
    'Meta-Llama-3-8B': [1, 2, 4, 8, 16, 32],
    'Meta-Llama-3-70B': [1, 2, 4, 8, 16, 24, 32],
    'Yi-34B-200K': [1, 2, 4, 8, 16]
}

titles = {
    'Yi-6B-200K': 'Yi-6B (1 A100)',
    'Meta-Llama-3-8B': 'LLaMA-3-8B (2 A100s, TP-2)',
    'Meta-Llama-3-70B': 'LLaMA-3-70B (8 H100s, TP-8)',
    'Yi-34B-200K': 'Yi-34B (2 A100s, TP-2)'
}

"""
def get_baseline_throughput(df, model, bs):
    df_model = df[(df['model'] == model) & (df['ptokens'] == context_length)]
    return df_model[(df_model['attn'] == 'vLLM') & (df_model['bs'] == bs)]['throughput'].values[0]
"""

def get_metric_value(df, attn, metric):
    if metric == 'throughput':
        return df[df['attn'] == attn]['throughput']
    elif metric == 'gpu_attention':
        return df[df['attn'] == attn]['attn_latency']
    elif metric == 'gpu_others':
        return df[df['attn'] == attn]['model_latency'] - df[df['attn'] == attn]['attn_latency']
    elif metric == 'cpu':
        return df[df['attn'] == attn]['cpu_latency']
    #elif metric == 'cpu':
    #    return df[df['attn'] == attn]['batch_latency'] - df[df['attn'] == attn]['model_latency']
    else:
        raise ValueError(f"Invalid metric: {metric}")

def get_y_max_and_ticks(model, metric):
    if metric == 'throughput':
        return np.arange(0, 310, 30) if model == 'Yi-34B-200K' else np.arange(0, 760, 75)
    elif metric == 'gpu_attention':
        return np.arange(0, 102, 10) if model == 'Yi-34B-200K' else np.arange(0, 72, 10)
    elif metric == 'gpu_others':
        return np.arange(0, 32, 5)
    elif metric == 'cpu':
        return np.arange(0, 8, 1) if model == 'Yi-6B-200K' else np.arange(0, 13, 2) if model == 'Meta-Llama-3-8B' else np.arange(0, 22, 3)
    else:
        raise ValueError(f"Invalid metric: {metric}")

def plot_common(df, model1, model2, model3, metric, y_label, plot_name):
    fig, axs = plt.subplots(ncols=3, figsize=(36, 8))
    width = 0.15

    models = [model1, model2, model3]
    for ax_index, model in enumerate(models):
        ax = axs[ax_index]
        df_model = df[(df['model'] == model) & (df['ptokens'] == context_length)]
        batch_sizes = batchsizes[model]
        x_pos = np.arange(len(batch_sizes))

        for i, bs in enumerate(batch_sizes):
            df_curr = df_model[df_model['bs'] == bs]
            ax.bar(i - 2.5*width, get_metric_value(df_curr, 'vLLM_16', metric), width, color=colors['vLLM_16'], alpha=opacity, hatch=hatches['vLLM_16'], edgecolor='black')
            ax.bar(i - 1.5*width, get_metric_value(df_curr, 'vLLM_128', metric), width, color=colors['vLLM_128'], alpha=opacity, hatch=hatches['vLLM_128'], edgecolor='black')
            ax.bar(i - 0.5*width, get_metric_value(df_curr, 'FLASH_PAGED_16', metric), width, color=colors['FLASH_PAGED_16'], alpha=opacity, hatch=hatches['FLASH_PAGED_16'], edgecolor='black')
            ax.bar(i + 0.5*width, get_metric_value(df_curr, 'FLASH_PAGED_128', metric), width, color=colors['FLASH_PAGED_128'], alpha=opacity, hatch=hatches['FLASH_PAGED_128'], edgecolor='black')
            ax.bar(i + 1.5*width, get_metric_value(df_curr, 'FLASH_vATTN_v2', metric), width, color=colors['FLASH_vATTN_v2'], alpha=1, hatch=hatches['FLASH_vATTN_v2'], edgecolor='black')

        x_ticks = [f'{bs}*16K' for bs in batch_sizes]
        ax.set_xticks(x_pos)
        ax.set_xticklabels(x_ticks, fontsize=20)
        #y_max, y_ticks = get_y_max_and_ticks(metric)
        #ax.set_yticks(np.arange(0, y_max, y_ticks))
        ax.set_yticks(get_y_max_and_ticks(model, metric))
        ax.tick_params(labelsize=24)
        ax.grid(axis='y', linestyle='--')
        ax.set_xlabel('Batch Size * Context Length', fontsize=24)
        ax.set_title(titles[model], fontweight='bold')
        if ax_index == 0:
            ax.set_ylabel(y_label, fontweight='bold', fontsize=28)
            if metric == 'throughput':
                ax.legend(legend, loc='upper left', fontsize='28', ncols=1)

    # Create a common legend for both subplots
    #handles, labels = axs[0].get_legend_handles_labels()
    #fig.legend(handles, legend, loc='upper center', ncol=2, frameon=True, fontsize=32)
    #fig.legend(legend, loc='upper center', fontsize='36', ncols=4)
    plt.tight_layout()
    #fig.subplots_adjust(top=0.8)
    plt.savefig(plot_name)


def get_combined_df():
    df = None
    for model in models:
        tmp = pd.read_csv(f"{model}/data.csv", sep=";")
        df = tmp if df is None else pd.concat([df, tmp])
    return df

df = get_combined_df()
plot_common(df, models[0], models[1], models[2], 'throughput', 'Decode Throughput', 'mean_decode_throughput.pdf')
plot_common(df, models[0], models[1], models[2], 'gpu_attention', 'Attention Time (GPU, ms)', 'mean_gpu_attn_latency.pdf')
plot_common(df, models[0], models[1], models[2], 'gpu_others', 'GPU Time w/o Attention (ms)', 'mean_gpu_others_latency.pdf')
plot_common(df, models[0], models[1], models[2], 'cpu', 'Mean CPU Time (ms)', 'mean_cpu_latency.pdf')
