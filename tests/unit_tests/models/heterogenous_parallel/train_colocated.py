"""
Colocated Heterogeneous Parallel Training for MIMO Models

This script trains MIMO models where encoder and LLM are colocated (same GPUs)
but have different TP/DP configurations, using the forward_backward_no_pipelining schedule.

Usage:
    uv run python -m torch.distributed.run --nproc_per_node=8 \
        tests/unit_tests/models/heterogenous_parallel/train_colocated.py
"""

import torch
import torch.distributed as dist
from functools import partial
import logging
import os
from typing import Any, Dict, Optional

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', force=True)

from tests.unit_tests.test_utilities import Utils
from tests.unit_tests.models.heterogenous_parallel.model_specs import get_vlm_mimo_model_colocated
from tests.unit_tests.models.heterogenous_parallel.parallel_utils import (
    get_module_to_grid_tuple,
    multimodule_no_sync,
    finalize_model_grads,
    zero_grad_buffer_for_multimodule,
)
from tests.unit_tests.models.heterogenous_parallel.dp_aware_data_iterator import (
    get_data_iterator_colocated,
    get_batch,
)
from tests.unit_tests.models.heterogenous_parallel.performance_utils import create_performance_monitor
from tests.unit_tests.models.heterogenous_parallel.config import (
    ModelConfig, ModuleArchConfig, ModuleParallelismConfig,
    DataConfig, RuntimeConfig
)
from tests.unit_tests.pipeline_parallel.test_multimodule_schedules import (
    _get_pg_collection_with_embedding_groups,
)
import megatron.core.pipeline_parallel.schedules as schedule


def loss_func(loss_mask, output_tensor):
    """Simple loss function for MIMO model training."""
    losses = output_tensor.float()
    loss_mask = loss_mask.contiguous().view(-1).float()
    
    total_tokens = loss_mask.sum().clone().detach().to(torch.int)
    total_loss = torch.sum(losses.view(-1) * loss_mask)
    reporting_loss = torch.cat([total_loss.clone().detach().view(1), total_tokens.view(1)])
    
    return (total_loss, total_tokens, {'lm loss': reporting_loss})


def _slice_batch_dim(data: Any, start: int, size: int, batch_dim: int = 0) -> Any:
    """Recursively slice tensors along batch dimension."""
    if isinstance(data, torch.Tensor):
        return data.narrow(batch_dim, start, size).contiguous()
    elif isinstance(data, dict):
        return {k: _slice_batch_dim(v, start, size, batch_dim) for k, v in data.items()}
    elif isinstance(data, (list, tuple)):
        return type(data)(_slice_batch_dim(v, start, size, batch_dim) for v in data)
    return data


def forward_step(data_iterator, model, encoder_grid=None, llm_grid=None):
    """Forward step for colocated MIMO model training.
    
    Handles data slicing for heterogeneous DP configurations:
    - Fan-in (encoder_dp > llm_dp): slice modality_inputs for encoder
    - Fan-out (encoder_dp < llm_dp): slice input_ids/labels/loss_mask for LLM
    
    Args:
        data_iterator: Data iterator
        model: MIMO model
        encoder_grid: HyperCommGrid for encoder module
        llm_grid: HyperCommGrid for LLM module
    """
    data_batch = get_batch(data_iterator)
    if data_batch is None:
        data_batch = {'input_ids': None}
        output_tensor, loss_mask = model(**data_batch)
        return output_tensor, partial(loss_func, loss_mask)
    
    # If grids not provided, no slicing needed (homogeneous case)
    if encoder_grid is None or llm_grid is None:
        output_tensor, loss_mask = model(**data_batch)
        return output_tensor, partial(loss_func, loss_mask)
    
    encoder_dp = encoder_grid.get_pg("dp").size()
    llm_dp = llm_grid.get_pg("dp").size()
    
    if encoder_dp > llm_dp:
        # Fan-in: data loaded with LLM DP (larger batch)
        # Slice modality_inputs for encoder's smaller batch
        scale = encoder_dp // llm_dp
        encoder_dp_idx = encoder_grid.get_pg("dp").rank()
        slot = encoder_dp_idx % scale
        
        batch_size = data_batch['input_ids'].shape[0]
        slice_size = batch_size // scale
        start = slot * slice_size
        
        # Slice modality_inputs for encoder
        # Note: modality_inputs tensors are [seq, batch, hidden], so batch_dim=1
        if 'modality_inputs' in data_batch and data_batch['modality_inputs'] is not None:
            data_batch['modality_inputs'] = _slice_batch_dim(
                data_batch['modality_inputs'], start, slice_size, batch_dim=1
            )
            logging.debug(
                f"[Rank {dist.get_rank()}] Fan-in: sliced modality_inputs "
                f"slot={slot}, scale={scale}, start={start}, size={slice_size}"
            )
    
    elif llm_dp > encoder_dp:
        # Fan-out: data loaded with encoder DP (larger batch)
        # Slice input_ids/labels/loss_mask for LLM's smaller batch
        scale = llm_dp // encoder_dp
        llm_dp_idx = llm_grid.get_pg("dp").rank()
        slot = llm_dp_idx % scale
        
        batch_size = data_batch['input_ids'].shape[0]
        slice_size = batch_size // scale
        start = slot * slice_size
        
        # Slice LLM inputs
        for key in ['input_ids', 'labels', 'loss_mask', 'position_ids']:
            if key in data_batch and data_batch[key] is not None:
                data_batch[key] = _slice_batch_dim(data_batch[key], start, slice_size)
        
        logging.debug(
            f"[Rank {dist.get_rank()}] Fan-out: sliced LLM inputs "
            f"slot={slot}, scale={scale}, start={start}, size={slice_size}"
        )
    
    # Equal DP: no slicing needed
    
    output_tensor, loss_mask = model(**data_batch)
    return output_tensor, partial(loss_func, loss_mask)


def train_colocated_mimo(
    model_config: ModelConfig,
    data_config: DataConfig,
    runtime_config: RuntimeConfig,
):
    """Train MIMO model with colocated heterogeneous parallelism.
    
    Uses forward_backward_no_pipelining schedule with ColocatedBridgeCommunicator
    for cross-module communication.
    
    Args:
        model_config: Model configuration (architectures, parallelisms)
        data_config: Data configuration (batch sizes, dataset params)
        runtime_config: Runtime configuration (iterations, profiling, etc)
    """
    logging.info("Creating colocated VLM MIMO model...")
    
    # Create MIMO model with colocated communication
    mimo_model, module_to_grid_map, topology = get_vlm_mimo_model_colocated(
        model_config=model_config,
        seq_len=data_config.seq_length,
    )
    
    mimo_model.config.barrier_with_L1_time = False
    
    logging.info(f"Rank {dist.get_rank()}: Model created with colocated communicators")
    
    # Log communicator info
    if mimo_model.colocated_comms:
        for (src, dest), comm in mimo_model.colocated_comms.items():
            logging.info(
                f"Rank {dist.get_rank()}: Communicator {src} -> {dest}: "
                f"scale_factor={comm.dp_scale_factor}"
            )
    
    # Set up module to grid tuple for no_sync and finalize_model_grads
    encoder_grid = module_to_grid_map[model_config.encoder_module_name]
    llm_grid = module_to_grid_map[model_config.llm_module_name]
    
    module_to_grid_tuple = get_module_to_grid_tuple(
        mimo_model, encoder_grid, llm_grid
    )
    
    # Configure no_sync and finalize_model_grads functions
    mimo_model.config.no_sync_func = partial(
        multimodule_no_sync,
        module_to_grid_tuple=module_to_grid_tuple
    )
    mimo_model.config.finalize_model_grads_func = partial(
        finalize_model_grads,
        module_to_grid_tuple=module_to_grid_tuple
    )
    
    logging.info(f"Rank {dist.get_rank()}: Creating colocated data iterator...")
    
    # Get data iterator using encoder's DP
    data_iterator = get_data_iterator_colocated(
        model_config=model_config,
        data_config=data_config,
        module_to_grid_map=module_to_grid_map,
    )
    
    # Create performance monitor
    perf_monitor = create_performance_monitor(
        model_config=model_config,
        data_config=data_config,
        runtime_config=runtime_config,
        megatron_config=mimo_model.config,
    )
    
    logging.info(f"Rank {dist.get_rank()}: Performance monitor initialized")
    
    # Get pg_collection for the schedule (use LLM's since it has embeddings)
    pg_collection = _get_pg_collection_with_embedding_groups(llm_grid)
    
    # Set model type for unit test
    mimo_model.model_type = 'unit-test'
    
    # Create forward_step_func with grids bound via partial
    forward_step_func = partial(
        forward_step,
        encoder_grid=encoder_grid,
        llm_grid=llm_grid,
    )
    
    # Prepare common arguments for schedule
    common_args = {
        'forward_step_func': forward_step_func,
        'data_iterator': data_iterator,
        'model': [mimo_model],
        'num_microbatches': data_config.num_microbatches,
        'seq_length': data_config.seq_length,
        'micro_batch_size': data_config.base_batch_size,
        'forward_only': False,
        'pg_collection': pg_collection,
    }
    
    all_losses = []
    
    for iteration in range(runtime_config.num_iterations):
        # Start iteration timing
        perf_monitor.start_iteration()
        
        # Run forward_backward_no_pipelining schedule
        losses_reduced = schedule.forward_backward_no_pipelining(**common_args)
        
        all_losses.append(losses_reduced)
        
        # End iteration timing
        perf_monitor.end_iteration()
        
        # Log performance metrics
        perf_monitor.log_performance(iteration=iteration + 1)
        
        # Zero grad buffers for next iteration
        zero_grad_buffer_for_multimodule(module_to_grid_tuple)
    
    # Save metrics
    metrics_dir = runtime_config.metrics_output_dir
    os.makedirs(metrics_dir, exist_ok=True)
    
    vision_arch = model_config.get_arch(model_config.encoder_module_name)
    llm_arch = model_config.get_arch(model_config.llm_module_name)
    vision_parallel = model_config.get_parallelism(model_config.encoder_module_name)
    llm_parallel = model_config.get_parallelism(model_config.llm_module_name)
    
    metrics_file = os.path.join(
        metrics_dir,
        f"colocated_vtp{vision_parallel.tensor_parallel}_ltp{llm_parallel.tensor_parallel}_"
        f"vdp{vision_parallel.data_parallel}_ldp{llm_parallel.data_parallel}_"
        f"mb{data_config.num_microbatches}.json"
    )
    perf_monitor.save_metrics_to_file(
        filepath=metrics_file,
        extra_info={
            'exp_name': 'colocated_mimo_training',
            'vision_tp': vision_parallel.tensor_parallel,
            'vision_dp': vision_parallel.data_parallel,
            'llm_tp': llm_parallel.tensor_parallel,
            'llm_dp': llm_parallel.data_parallel,
        },
        exclude_warmup=True
    )
    
    logging.info(f"Rank {dist.get_rank()}: Colocated training completed.")
    
    # Cleanup: destroy process groups to free GPU memory for next experiment
    for grid in module_to_grid_map.values():
        if hasattr(grid, '_pgs'):
            for pg in grid._pgs.values():
                if pg is not None:
                    try:
                        dist.destroy_process_group(pg)
                    except:
                        pass
            grid._pgs.clear()
    
    # Delete model to free GPU memory
    del mimo_model
    
    return all_losses


if __name__ == "__main__":
    # Initialize distributed training
    Utils.initialize_distributed()
    
    # Create configurations for colocated heterogeneous training
    # Example: Vision TP2/DP4, LLM TP4/DP2 on 8 GPUs
    model_config = ModelConfig(
        module_architectures={
            'images': ModuleArchConfig(
                num_layers=2,
                hidden_size=256,
                num_attention_heads=8,
                seq_length=256,
                vocab_size=0,  # Vision encoder has no vocab
            ),
            'language_module': ModuleArchConfig(
                num_layers=2,
                hidden_size=256,
                num_attention_heads=8,
                seq_length=1024,
                vocab_size=4000,
            ),
        },
        # Equal DP case: same TP/DP for both
        module_parallelisms={
            'images': ModuleParallelismConfig(
                tensor_parallel=8,
                pipeline_parallel=1,
                data_parallel=1,
            ),
            'language_module': ModuleParallelismConfig(
                tensor_parallel=1,
                pipeline_parallel=1,
                data_parallel=8,
            ),
        },
        special_token_ids={'images': 32000},
        llm_module_name='language_module',
        llm_rank_offset=0,  # 0 = colocated
    )
    
    data_config = DataConfig(
        base_batch_size=4,  # Per LLM DP replica
        num_microbatches=4,
        seq_length=1024,
        image_seq_length=256,
        vocab_size=4000,
        image_special_token_id=32000,
        dataset_size=1024,
        num_workers=0,  # Set to 0 for simpler debugging
        prefetch_factor=None,
        pin_memory=False,
        persistent_workers=False,
    )
    
    runtime_config = RuntimeConfig(
        num_iterations=4,
        warmup_iterations=1,
        log_interval=1,
        enable_performance_monitoring=True,
        metrics_output_dir=os.environ.get("METRICS_OUTPUT_DIR", "./logs/metrics"),
        pipeline_schedule="colocated",
        enable_profiling=False,
    )
    
    # Run training
    losses = train_colocated_mimo(
        model_config=model_config,
        data_config=data_config,
        runtime_config=runtime_config,
    )
    
    logging.info(f"Rank {dist.get_rank()}: Final losses count = {len(losses)}")
    
    dist.destroy_process_group()
