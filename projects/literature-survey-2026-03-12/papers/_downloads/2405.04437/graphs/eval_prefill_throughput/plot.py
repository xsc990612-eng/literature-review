import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd

plt.rcParams.update({'font.size': 24})
plt.rcParams.update({'font.family': 'Sans Serif'})

#configs = ["FlashAttention", "FlashInfer", "vAttention"]
configs = ["FA_Paged", "FI_Paged", "FA_vAttention", "FI_vAttention"]

config_labels = {
    'FA_Paged': 'FA2_Paged',
    'FI_Paged': 'FI_Paged',
    'FA_vAttention': 'FA2_vAttention',
    'FI_vAttention': 'FI_vAttention',
}

colors = {
    'FA_Paged': 'chocolate',
    'FI_Paged': 'green',
    'FA_vAttention': 'chocolate',
    'FI_vAttention': 'green',
}

linestyles = {
    'FA_Paged': '-',
    'FI_Paged': '-',
    'FA_vAttention': '--',
    'FI_vAttention': '--',
}

markerstyles = {
    'FA_Paged': 'o',
    'FI_Paged': 's',
    'FA_vAttention': 'D',
    'FI_vAttention': '^'
}

context_lengths = [1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 196608]

def plot_throughput(df, figname, title):
    plt.figure(figsize=(10, 5))
    for i, cfg in enumerate(configs):
        plt.plot(df['cl'], df[cfg], label=config_labels[cfg], color=colors[cfg], linestyle = linestyles[cfg], linewidth=2, marker=markerstyles[cfg], markersize=7)
    
    plt.xlabel('Context Length', fontsize=24, fontweight='bold')
    plt.ylabel('Tokens/second', fontsize=24, fontweight='bold')
    plt.xticks(fontsize=18)
    if 'llama-3' in figname:
        plt.yticks(np.arange(0, 24100, 4000))
    elif 'yi-34b' in figname:
        plt.yticks(np.arange(0, 5100, 1000))
    else: 
        plt.yticks(np.arange(0, 16000, 3000))
    #plt.xticks(np.arange(0, 330, 64))
    plt.grid()
    plt.tight_layout()
    plt.title(title, fontsize=18, fontweight='bold')
    #if "llama" in figname:
    plt.legend(loc='lower left', ncols=1, fontsize=24)
    plt.savefig(figname)

def plot_latency(df, figname, title):
    plt.figure(figsize=(10, 5))
    for i, cfg in enumerate(configs):
        plt.plot(df['cl'], df[cfg], label=config_labels[cfg], color=colors[cfg], linestyle = linestyles[cfg], linewidth=2, marker=markerstyles[cfg], markersize=7)
    
    plt.xlabel('Context Length', fontsize=24, fontweight='bold')
    plt.ylabel('Latency (ms)', fontsize=24, fontweight='bold')
    plt.xticks(fontsize=18)
    if 'llama-3' in figname:
        plt.yticks(np.arange(0, 40, 5))
    elif 'yi-34b' in figname:
        plt.yticks(np.arange(0, 121, 20))
    else: 
        plt.yticks(np.arange(0, 61, 10))
    #plt.xticks(np.arange(0, 330, 64))
    plt.grid()
    plt.tight_layout()
    plt.title(title, fontsize=18, fontweight='bold')
    #if "llama" in figname:
    plt.legend(loc='upper left', ncols=1, fontsize=24)
    plt.savefig(figname)

df_yi6b = pd.read_csv('Yi-6B/data.csv', sep=';')
df_llama3 = pd.read_csv('Llama-3-8B/data.csv', sep=';')
df_yi34b = pd.read_csv('Yi-34B/data.csv', sep=';')
plot_throughput(df_yi6b, 'yi-6b-prefill-throughput.pdf', 'Yi-6B (1 A100)')
plot_throughput(df_llama3, 'llama-3-8b-prefill-throughput.pdf', 'Llama-3-8B (2 A100s)')
plot_throughput(df_yi34b, 'yi-34b-prefill-throughput.pdf', 'Yi-34B (2 A100s)')

df_yi6b = pd.read_csv('Yi-6B/attention.csv', sep=';')
df_llama3 = pd.read_csv('Llama-3-8B/attention.csv', sep=';')
df_yi34b = pd.read_csv('Yi-34B/attention.csv', sep=';')
plot_latency(df_yi6b, 'yi-6b-prefill-attn-latency.pdf', 'Yi-6B (1 A100)')
plot_latency(df_llama3, 'llama-3-8b-prefill-attn-latency.pdf', 'Llama-3-8B (2 A100s)')
plot_latency(df_yi34b, 'yi-34b-prefill-attn-latency.pdf', 'Yi-34B (2 A100s)')
