import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd

plt.rcParams.update({'font.size': 24})
plt.rcParams.update({'font.family': 'Sans Serif'})

configs = ["vLLM", "FA_Paged", "FI_Paged", "FA_vAttention"]

colors = {
    'vLLM': 'chocolate',
    'FA_Paged': 'blue',
    'FI_Paged': 'red',
    'FA_vAttention': 'green',
    'FI_vAttention': 'green',
}

linestyles = {
    'vLLM': '-',
    'FA_Paged': '-',
    'FI_Paged': '-',
    'FA_vAttention': '--',
    'FI_vAttention': '--',
}

markerstyles = {
    'vLLM': '^',
    'FA_Paged': 'o',
    'FI_Paged': 's',
    'FA_vAttention': 'D',
    'FI_vAttention': '^'
}

legends = {
    'vLLM': 'vLLM',
    'FA_Paged': 'FA2_Paged',
    'FI_Paged': 'FI_Paged',
    'FA_vAttention': 'FA2_vAttention',
    'FI_vAttention': 'FI_vAttention',
}

context_lengths = [1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 196608]

def plot_throughput(df, figname, title):
    plt.figure(figsize=(10, 5))
    x_ticks = np.arange(len(df['bs']))
    for i, cfg in enumerate(configs):
        plt.plot(x_ticks, df[cfg], label=legends[cfg], color=colors[cfg], linestyle = linestyles[cfg], linewidth=2, marker=markerstyles[cfg], markersize=7)
    
    plt.xlabel('Batch Size', fontsize=24, fontweight='bold')
    plt.ylabel('Tokens/second', fontsize=24, fontweight='bold')
    x_labels = [str(int(x)) for x in df['bs']]
    plt.xticks(ticks=x_ticks, labels=x_labels, fontsize=24)
    if 'llama-3' in figname:
        plt.yticks(np.arange(0, 1100, 200))
    elif 'yi-34b' in figname:
        plt.yticks(np.arange(0, 500, 100))
    else: 
        plt.yticks(np.arange(0, 1100, 200))
    plt.grid()
    plt.tight_layout()
    plt.title(title, fontsize=24, fontweight='bold')
    plt.legend(loc='upper left', ncols=2, fontsize=24)
    plt.savefig(figname)

df_yi6b = pd.read_csv('Yi-6B/data.csv', sep=';')
df_llama3 = pd.read_csv('Llama-3-8B/data.csv', sep=';')
df_yi34b = pd.read_csv('Yi-34B/data.csv', sep=';')
plot_throughput(df_yi6b, 'yi-6b-decode-throughput.pdf', 'Yi-6B (1 A100)')
plot_throughput(df_llama3, 'llama-3-8b-decode-throughput.pdf', 'Llama-3-8B (2 A100s)')
plot_throughput(df_yi34b, 'yi-34b-decode-throughput.pdf', 'Yi-34B (2 A100s)')
