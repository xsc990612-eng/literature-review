import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd

plt.rcParams.update({'font.size': 20})
plt.rcParams.update({'font.family': 'Sans Serif'})

colors = ['chocolate', 'green', 'blue', 'gray', 'lightgreen']
line_styles = ['-', '--', ':', '-.']

models = ['Yi-6B-200K', 'Meta-Llama-3-8B' , 'Yi-34B-200K']
#models = ['Meta-Llama-3-8B' , 'Yi-34B-200K']

labels = {
    'Yi-6B-200K': 'Yi-6B',
    'Meta-Llama-3-8B': 'Llama-3-8B',
    'Yi-34B-200K': 'Yi-34B',
}

kv_sizes = {
    'Yi-6B-200K': 64,
    'Meta-Llama-3-8B': 128,
    'Yi-34B-200K': 240,
}

def get_yticks(metric):
    if metric == 'throughput':
        return np.arange(0, 6100, 1000)
    elif metric == 'batch_latency':
        return np.arange(0, 145, 20)
    else:
        raise ValueError(f'Invalid metric: {metric}')

batch_sizes = [1, 2, 4, 8, 16, 32, 64, 96, 128, 160, 192, 224, 256, 288, 320]

def plot_perf_metric(df, metric, figname, ylabel):
    plt.figure(figsize=(7, 5))
    for i, model in enumerate(models):
        model_df = df[df['model'] == model]
        plt.plot(model_df['bs'], model_df[metric], label=labels[model], color=colors[i], linestyle = line_styles[i], linewidth=3)
    
    plt.xlabel('Batch Size', fontweight='bold', fontsize=22)
    plt.ylabel(ylabel, fontweight='bold', fontsize=22)
    plt.yticks(get_yticks(metric))
    plt.xticks(np.arange(0, 330, 64))
    plt.grid()
    plt.tight_layout()
    #if metric == 'throughput':
    plt.legend(loc='lower right', ncols=1, fontsize=20)
    plt.savefig(figname)

def plot_allocattion_rate(df):
    plt.figure(figsize=(7, 5))
    for i, model in enumerate(models):
        model_df = df[df['model'] == model]
        df_bs = model_df['bs']
        df_throughput = model_df['throughput'] * (kv_sizes[model]/1024)
        plt.plot(df_bs, df_throughput, label=labels[model], color=colors[i], linestyle = line_styles[i], linewidth=3)
    
    plt.xlabel('Batch Size', fontweight='bold', fontsize=22)
    plt.ylabel('Memory Allocatation \n (MB per second)', fontweight='bold', fontsize=22)
    plt.yticks(np.arange(0, 610, 100))
    plt.xticks(np.arange(0, 330, 64))
    plt.grid()
    plt.tight_layout()
    plt.legend(loc='lower right', ncols=1, fontsize=20)
    plt.savefig(f'analysis_allocation_rate.pdf')

df = pd.read_csv('data.csv', sep=';')
df_tput = df.groupby(['model', 'bs'])['throughput'].mean().reset_index()
df_latency = df.groupby(['model', 'bs'])['batch_latency'].mean().reset_index()
plot_perf_metric(df_tput, 'throughput', 'analysis_throughput.pdf', 'Tokens/second')
plot_perf_metric(df_latency, 'batch_latency', 'analysis_latency.pdf', 'Decode Latency (ms)')
plot_allocattion_rate(df_tput)