import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({'font.size': 40})
plt.rcParams.update({'font.family': 'Sans Serif'})

def plot_fig(df):
    plt.figure(figsize=(18, 8))
    df['x_values'] = range(0, len(df))
    plt.plot(df['x_values'], df['sync'], label='Without overlapping', color='red', linestyle='dashed', linewidth=2)
    #async_df['x_values'] = range(0, len(async_df))
    plt.plot(df['x_values'], df['async'], label='With overlapping', color='green', linestyle='-', linewidth=2)
    plt.yticks(np.arange(0, 61, 10))
    plt.grid()
    plt.ylabel('Latency (ms)', fontsize=36, fontweight='bold')
    plt.xlabel('Decode Iteration', fontsize=36, fontweight='bold')
    plt.legend(loc='lower left', fontsize=40)
    #plt.title('Llama-3-8B (1 A100)', fontsize=32, fontweight='bold')
    plt.tight_layout()
    plt.savefig('latency.pdf')

df = pd.read_csv('data.csv', sep='\t')
df['sync'] = df['sync'].apply(lambda x: x * 1000)
df['async'] = df['async'].apply(lambda x: x * 1000)
plot_fig(df)
