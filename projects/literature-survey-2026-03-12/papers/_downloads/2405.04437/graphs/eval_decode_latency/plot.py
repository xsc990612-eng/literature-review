import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({'font.size': 40})
plt.rcParams.update({'font.family': 'Sans Serif'})

curr = os.path.dirname(os.path.realpath(__file__))
sync_path = 'sync_bs_4/batch_metrics.csv'
async_path = 'async_bs_4/batch_metrics.csv'

sync_df = pd.read_csv(os.path.join(curr, sync_path))
sync_df = sync_df[sync_df['batch_num_decode_tokens'] > 0]*1000

async_df = pd.read_csv(os.path.join(curr, async_path))
async_df = async_df[async_df['batch_num_decode_tokens'] > 0]*1000

def plot_fig(sync_df, async_df):
    plt.figure(figsize=(15, 8))
    sync_df['x_values'] = range(0, len(sync_df))
    plt.plot(sync_df['x_values'], sync_df['batch_execution_time'], label='Without overlapping', color='red', linestyle='-', linewidth=2)
    async_df['x_values'] = range(0, len(async_df))
    plt.plot(async_df['x_values'], async_df['batch_execution_time'], label='With overlapping', color='green', linestyle='-', linewidth=2)
    plt.yticks(np.arange(0, 51, 5))
    plt.grid()
    plt.ylabel('Latency (ms)', fontsize=40, fontweight='bold')
    plt.xlabel('Decode Iteration', fontsize=40, fontweight='bold')
    plt.legend(loc='lower right', fontsize=40)
    #plt.title('Llama-3-8B (1 A100)', fontsize=32, fontweight='bold')
    plt.tight_layout()
    plt.savefig('latency.pdf')

plot_fig(sync_df, async_df)
