MEMORY_MARGIN_FRACTION=0.1

# NOTE: ASSUMING PIPELINE_STAGES=1
PIPELINE_STAGES=1

model_config = {}
model_config["llama-7b"] = {
    "TP_WORKERS": 1,
    "REQ_MAX_TOKENS": 4096,
    "GATED_MLP": True,
    "NUM_LAYERS": 32,
    "Q_HEADS": 32,
    "KV_HEADS": 32,
    "EMBEDDING_DIM": 4096,
    "MLP_HIDDEN_DIM": 11008,
    "TOTAL_MEM_GB":80,
}

model_config["yi-34b"] = {
    "TP_WORKERS": 2,
    "REQ_MAX_TOKENS": 8192,
    "GATED_MLP": True,
    "NUM_LAYERS": 60,
    "Q_HEADS": 56,
    "KV_HEADS": 8,
    "EMBEDDING_DIM": 7168,
    "MLP_HIDDEN_DIM": 20480,
    "TOTAL_MEM_GB":160,
}

# MODEL SPECIFIC contents (for now Llama-7B)
def get_model_param(model, param):
    if param == "ATTN_HEAD_DIM":
        return model_config[model]["EMBEDDING_DIM"] // model_config[model]["Q_HEADS"]
    if param == "Q_HEADS_PER_TENSOR_PARALLEL_WORKER":
        return model_config[model]["Q_HEADS"] // model_config[model]["TP_WORKERS"]
    if param == "KV_HEADS_PER_TENSOR_PARALLEL_WORKER":
        return model_config[model]["KV_HEADS"] // model_config[model]["TP_WORKERS"]
    return model_config[model][param]

# MODEL SPECIFIC contents (for now Yi-34B)
# TP_WORKERS=2
# REQ_MAX_TOKENS=8192
# GATED_MLP=True
# NUM_LAYERS=60
# Q_HEADS=56
# KV_HEADS=8
# EMBEDDING_DIM=7168
# MLP_HIDDEN_DIM=20480

#Q_HEADS_PER_TENSOR_PARALLEL_WORKER=(Q_HEADS // TP_WORKERS)
#ATTN_HEAD_DIM=(EMBEDDING_DIM // Q_HEADS)
#KV_HEADS_PER_TENSOR_PARALLEL_WORKER=(KV_HEADS // TP_WORKERS)

def get_num_parameters_per_layer(model):
    EMBEDDING_DIM = get_model_param(model, "EMBEDDING_DIM")
    ATTN_HEAD_DIM = get_model_param(model, "ATTN_HEAD_DIM")
    Q_HEADS_PER_TENSOR_PARALLEL_WORKER = get_model_param(model, "Q_HEADS_PER_TENSOR_PARALLEL_WORKER")
    KV_HEADS_PER_TENSOR_PARALLEL_WORKER = get_model_param(model, "KV_HEADS_PER_TENSOR_PARALLEL_WORKER")
    MLP_HIDDEN_DIM = get_model_param(model, "MLP_HIDDEN_DIM")
    TP_WORKERS = get_model_param(model, "TP_WORKERS")
    GATED_MLP = get_model_param(model, "GATED_MLP")
    num_parameters = 0
    # weights for attention metrics Wq, Wk, Wv
    num_parameters += (
        EMBEDDING_DIM * ATTN_HEAD_DIM
        * (
            Q_HEADS_PER_TENSOR_PARALLEL_WORKER + 2 * KV_HEADS_PER_TENSOR_PARALLEL_WORKER
        )
    )
    # weights for attention metrics Wo
    num_parameters += (
        EMBEDDING_DIM * ATTN_HEAD_DIM *
        Q_HEADS_PER_TENSOR_PARALLEL_WORKER
    )
    # fc layer weights
    if GATED_MLP:
        num_parameters += (
            3 * EMBEDDING_DIM * MLP_HIDDEN_DIM // TP_WORKERS
        )
    else:
        num_parameters += (
            2 * EMBEDDING_DIM * MLP_HIDDEN_DIM // TP_WORKERS
        )

    return num_parameters

def get_num_parameters_per_device(model):
    return get_num_parameters_per_layer(model) * get_model_param(model, "NUM_LAYERS")

def get_parameter_memory_per_device(model):
    return 2 * get_num_parameters_per_device(model)

def get_kv_cache_memory_per_layer_per_request(model, tokens):
    ATTN_HEAD_DIM = get_model_param(model, "ATTN_HEAD_DIM")
    KV_HEADS_PER_TENSOR_PARALLEL_WORKER = get_model_param(model, "KV_HEADS_PER_TENSOR_PARALLEL_WORKER")
    return (
        2  # 2 bytes per float
        * 2  # one for key, one for value
        * ATTN_HEAD_DIM
        * KV_HEADS_PER_TENSOR_PARALLEL_WORKER
        * tokens
    )

def get_kv_cache_memory_per_device_per_request(model, tokens):
    return get_kv_cache_memory_per_layer_per_request(model, tokens) * get_model_param(model, "NUM_LAYERS")

def get_max_batch_size(model) -> int:
    available_memory = (
        get_model_param(model, "TOTAL_MEM_GB") * 1024**3 * (1 - MEMORY_MARGIN_FRACTION)
    )
    parameter_memory_per_device = get_parameter_memory_per_device(model)
    kv_cache_memory_per_device_per_request = (
        get_kv_cache_memory_per_device_per_request(model, "REQ_MAX_TOKENS")
    )

    memory_for_kv_cache = available_memory - parameter_memory_per_device
    number_of_requests = (
        memory_for_kv_cache // kv_cache_memory_per_device_per_request
    )
    return number_of_requests

def get_max_blocks_per_sequence(model, block_size):
    return (get_model_param(model, "REQ_MAX_TOKENS") // block_size)

def get_max_request_slots(model):
    return get_max_batch_size() * get_model_param(model, "PIPELINE_STAGES")

def get_total_blocks(model, block_size):
    return get_max_blocks_per_sequence(model, block_size) * get_max_request_slots(model)

def get_model_weight_bytes(model):
    return get_parameter_memory_per_device(model)

def get_request_size_bytes(model, num_tokens):
    return get_kv_cache_memory_per_device_per_request(model, num_tokens)

def get_request_block_memory_usage(model, num_blocks, block_size):
    return get_request_size_bytes(model, num_blocks * block_size)
