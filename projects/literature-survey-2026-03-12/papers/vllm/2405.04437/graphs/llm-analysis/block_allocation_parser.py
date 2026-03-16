import argparse
from model_mem_usage import get_total_blocks, get_model_weight_bytes, get_request_size_bytes
from model_mem_usage import get_request_block_memory_usage
import matplotlib.pyplot as plt
import sys
import os
import numpy as np

INTER_REQ_DELIMITER="|"
INTRA_REQ_DELIMITER=","
trace_file = "trace.out"

parser = argparse.ArgumentParser()
parser.add_argument('--block_size', type=int, default=1)
parser.add_argument('--trace_file', type=str, default='trace.txt')
args = parser.parse_args()

trace_file=args.trace_file

def get_total_gpu_memoery(model):
    if model == "llama-7b":
        return 80
    if model == "yi-34b":
        return 160
    return 1

def gb_from_b(x):
    return round(x / (1024**3), 2)

def mb_from_b(x):
    return round(x / (1024**2), 2)

def get_block_ratio(x):
    return round((x / available_blocks) * 100, 2)

class Request:
    id: int
    num_processed_tokens: int
    num_prefill_tokens: int

    def __init__(self, line: str, block_size: int, model: str):
        parts = line.strip().split(INTRA_REQ_DELIMITER)
        self.id = int(parts[0])
        self.num_processed_tokens = int(parts[1])
        self.num_prefill_tokens = int(parts[2])
        self.block_size = block_size
        self.model = model
    
    def is_decode_iter(self):
        return self.num_processed_tokens > (self.num_prefill_tokens + 1)
    
    def iter_req_new_block(self):
        return self.block_size == 1 or self.num_processed_tokens % self.block_size == 1
    
    def get_prefill_blocks(self):
        return (self.num_prefill_tokens // self.block_size) + 1

    def get_total_blocks(self):
        return (self.num_processed_tokens // self.block_size) + 1
    
    def get_kv_cache_bytes(self):
        return get_request_size_bytes(self.model, self.num_processed_tokens)
    
# Get available blocks for KV-cache in the system
#available_blocks = get_total_blocks(BLOCK_SIZE)
def model_weight_bytes(model):
    return get_model_weight_bytes(model)

def process_trace(model, path, block_size):
    # This list tracks the number of new blocks allocated per iteration
    new_block_tracker = []
    # This list tracks the total number of blocks used at the end of iteration
    mem_util_tracker = []
    # This is 0/1 list (per iteration), where a 0 indicates no blocks were allocated
    # in that iteration, while a 1 indicates a block was allocated in that iteration
    block_frequency = []
    # This list tracks number of blocks or used in each iteration
    block_usage_tracker = []
    # A per request tracker dictionary for tracking request progress
    req_tracker = {}
    # tracking new memory allocations per iteration
    block_mem_alloc_tracker = []
    # track batch size in each iteration
    batch_size_tracker = []

    with open(path, "r") as f:
        for line in f:
            line = line.rstrip()
            new_blocks = 0
            new_blocks_mem = 0
            total_mem_bytes = 0
            total_blocks = 0
            global_kv_cache_size = 0

            if line.startswith('BATCH_RAW_INFO'):
                # This line has multiple batch information
                requests = line.split(INTER_REQ_DELIMITER)
                batch_size_tracker.append(len(requests) - 1)
                # 0th index is tracker, ignore
                for i in range(1, len(requests)):
                    req = Request(requests[i], block_size, model)
                    if req.id not in req_tracker:
                        # A prefill calculate blocks?
                        req_tracker[req.id] = req
                    # only update processed tokens
                    req_tracker[req.id].num_processed_tokens = req.num_processed_tokens
                    # DID IT REQUIRE NEW BLOCK ALLOCATION?
                    if req.is_decode_iter() and req.iter_req_new_block():
                            # A decode, update iteration_info and req_num_processed_tokens
                            new_blocks += 1
                            new_blocks_mem += get_request_size_bytes(model, block_size)
                    # Blocks used by the request
                    total_blocks += req.get_total_blocks()
                    global_kv_cache_size += get_request_block_memory_usage(model, req.get_total_blocks(), block_size)
            # account for lines in trace where there is no useful log
            if global_kv_cache_size > 0:
                new_block_tracker.append(new_blocks)
                block_mem_alloc_tracker.append(mb_from_b(new_blocks_mem))
                block_frequency.append(1 if new_blocks > 0 else 0)
                block_usage_tracker.append(total_blocks * block_size)

                global_mem_util = round(gb_from_b(global_kv_cache_size + model_weight_bytes(model)) / get_total_gpu_memoery(model), 2)
                mem_util_tracker.append(global_mem_util)

    #print(mem_util_tracker)
    return mem_util_tracker, block_mem_alloc_tracker

plt.rcParams.update({'font.size': 24})
plt.rcParams.update({'font.family': 'Sans Serif'})

root = os.path.dirname(os.path.realpath(__file__))
#llama_traces = ["llama-batch-0.8qps.txt", "llama-batch-1.0qps.txt", "llama-batch-1.2qps.txt"]
#yi_traces = ["yi-batch-0.8qps.txt", "yi-batch-1.0qps.txt", "yi-batch-1.2qps.txt"]
llama_traces = ["llama-7b-trace/llama-batch-1.2qps.txt"]
yi_traces = ["yi-34b-trace/yi34B-block-1.2.txt"]

hatches = ['', 'xx', '\\\\', '--']
colors = ['chocolate', 'wheat', 'cadetblue', 'gray', 'lightgreen']
opacity = [0.55, 0.75, 0.75, 0.75]
memcolors = ['gray', 'chocolate', 'cadetblue', 'green', 'lightgreen']

def plot_mem_util(model, mem_util_data):
    fig, ax = plt.subplots(figsize=(6, 5))
    labels = ['mean', 'p90', 'max']
    x = np.arange(len(labels))  # the label locations
    block_sizes = list(mem_util_data.keys())
    width = 0.3  if len(block_sizes) == 3 else 0.2
    positions = [-1, 0, 1] if len(block_sizes) == 3 else [-1.5, -.5, 0.5, 1.5]
    for i, bs in enumerate(block_sizes):
        mean = np.mean(mem_util_data[block_sizes[i]])
        p90 = np.percentile(mem_util_data[block_sizes[i]], 90)
        max = np.max(mem_util_data[block_sizes[i]])
        data = [mean, p90, max]
        plt.bar(x + width * positions[i], data, width, label=f'bs={bs}', color=colors[i], edgecolor='black', hatch=hatches[i], alpha=opacity[i])

    ax.set_ylabel('Memory Utilization', fontweight='bold', fontsize=22)
    ax.set_xticks(x)
    plt.ylim(0, 1)
    ax.yaxis.grid(True)
    ax.set_xticklabels(labels)
    ax.legend(fontsize=20)
    fig.tight_layout()
    plt.savefig(f'{model}_mem_util_tracker.pdf')

def plot_mem_alloc(model, mem_alloc_data):
    plt.figure(figsize=(6, 5))
    block_sizes = list(mem_alloc_data.keys())
    block_sizes_reversed = block_sizes[::-1]
    for i, block_size in enumerate(block_sizes_reversed):
        plt.plot(mem_alloc_data[block_size][1000:2001], label=f"bs={block_size}", color=memcolors[len(block_sizes) - (i+1)], linewidth=1)
    ticks = list(range(0, 1001, 250))
    labels = [str(tick+1000) for tick in ticks]
    plt.xticks(ticks, labels, fontsize=18)
    plt.xlabel('Iteration Number', fontweight='bold', fontsize=22)
    plt.ylabel('Allocation Size (MB)', fontweight='bold', fontsize=22)
    #y_max = 20 if bs == 1 else 50 if bs == 32 else 300
    y_max = 300
    plt.ylim(0, y_max)
    plt.grid()
    plt.tight_layout()
    plt.legend(loc='upper right', ncols=1, fontsize=20)
    plt.savefig(f'{model}_mem_alloc_tracker.pdf')


def plot_mem_alloc_cdf(model, mem_alloc_data):
    plt.figure(figsize=(6, 5))
    linestyles = ['-', '--', '-.', ':']
    markers = ['o', 's', 'x', 'D']
    block_sizes = list(mem_alloc_data.keys())
    for i, block_size in enumerate(block_sizes):
        data = mem_alloc_data[block_size][1000:2000]
        values, base = np.histogram(data, bins=40)
        cumulative = np.cumsum(values)
        plt.plot(base[:-1], cumulative / np.max(cumulative), label=f'bs={block_size}', linewidth=2, color=memcolors[i], linestyle=linestyles[i], marker=markers[i])
    plt.xlabel('Iteration Alloc Size (MB)', fontweight='bold', fontsize=22)
    plt.ylabel('CDF', fontweight='bold', fontsize=22)
    plt.ylim(0.2, 1)
    plt.xlim(0, 100)
    plt.grid()
    plt.tight_layout()
    plt.legend(loc='best', ncols=1, fontsize=20, frameon=True)
    plt.savefig(f'{model}_mem_alloc_cdf.pdf')

def process_model_traces(model):
    mem_util_trackers, mem_alloc_trackers = {}, {}
    trace = llama_traces[0] if model == "llama-7b" else yi_traces[0]
    block_sizes = [16, 32, 256] if model == "llama-7b" else [16, 32, 256, 2048]
    for block_size in block_sizes:
        trace_file = os.path.join(root, trace)
        mem_util_tracker, mem_alloc_tracker = process_trace(model, trace_file, block_size)
        mem_util_trackers[block_size] = mem_util_tracker
        mem_alloc_trackers[block_size] = mem_alloc_tracker

    #print(mem_util_trackers.keys())
    plot_mem_util(model, mem_util_trackers)
    plot_mem_alloc(model, mem_alloc_trackers)
    plot_mem_alloc_cdf(model, mem_alloc_trackers)

process_model_traces("llama-7b")
process_model_traces("yi-34b")

















sys.exit()
print(f"Block Allocation Frequency ({len(block_frequency)} iters) === === === ===")
block_frequency.sort()
print(f"Min       : {block_frequency[0]}")
print(f"Mean      : {sum(block_frequency) / len(block_frequency)}")
print(f"Median    : {block_frequency[len(block_frequency) // 2]}")
print(f"90th perc : {block_frequency[int(len(block_frequency) * 0.9)]}")
print(f"95th perc : {block_frequency[int(len(block_frequency) * 0.95)]}")
print(f"Max       : {block_frequency[-1]}")


print(f"Block Usage ({available_blocks} blocks available) === === === ===")
block_usage_tracker.sort()
print(f"Min       : {block_usage_tracker[0]}")
print(f"Mean      : {sum(block_usage_tracker) / len(block_usage_tracker)}")
print(f"Median    : {block_usage_tracker[len(block_usage_tracker) // 2]}")
print(f"90th perc : {block_usage_tracker[int(len(block_usage_tracker) * 0.9)]}")
print(f"95th perc : {block_usage_tracker[int(len(block_usage_tracker) * 0.95)]}")
print(f"Max       : {block_usage_tracker[-1]}")


print(f"Memory Utilization Tracker (in GBs) = === === ===")
mem_util_tracker.sort()
print(f"Min       : {mem_util_tracker[0]}")
print(f"Mean      : {sum(mem_util_tracker) / len(mem_util_tracker)}")
print(f"Median    : {mem_util_tracker[len(mem_util_tracker) // 2]}")
print(f"90th perc : {mem_util_tracker[int(len(mem_util_tracker) * 0.9)]}")
print(f"95th perc : {mem_util_tracker[int(len(mem_util_tracker) * 0.95)]}")
print(f"Max       : {mem_util_tracker[-1]}")
