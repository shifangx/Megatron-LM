"""
Training script for homogeneous parallelism (Case 1).

This script uses forward_backward_no_pipelining schedule when PP=1 for all modules.
Suitable for baseline experiments where vision and LLM use identical parallelism strategies.

Usage:
uv run python -m torch.distributed.run --nproc_per_node=8 tests/unit_tests/models/heterogenous_parallel/train_homogeneous.py
"""

import torch
import torch.distributed as dist
from functools import partial
import logging
import os

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', force=True)

from tests.unit_tests.test_utilities import Utils
from tests.unit_tests.models.heterogenous_parallel.model_specs import get_vlm_mimo_model_homogeneous
from tests.unit_tests.models.heterogenous_parallel.parallel_utils import (
    multimodule_no_sync, 
    finalize_model_grads,
    zero_grad_buffer_for_multimodule,
)
from tests.unit_tests.models.heterogenous_parallel.dp_aware_data_iterator import get_data_iterator, get_batch
from tests.unit_tests.models.heterogenous_parallel.performance_utils import create_performance_monitor
from tests.unit_tests.models.heterogenous_parallel.config import (
    ModelConfig, ModuleArchConfig, ModuleParallelismConfig,
    DataConfig, RuntimeConfig
)
import megatron.core.pipeline_parallel.schedules as schedule


def loss_func(loss_mask, output_tensor):
    """Simple loss function for MIMO model training.
    
    Args:
        loss_mask: mask indicating which tokens contribute to the loss
        output_tensor: model output tensor
    
    Returns:
        tuple: (loss, num_tokens, metrics_dict)
    """
    losses = output_tensor.float()
    
    loss_mask = loss_mask.contiguous().view(-1).float()
    
    total_tokens = loss_mask.sum().clone().detach().to(torch.int)
    total_loss = torch.sum(losses.view(-1) * loss_mask)
    reporting_loss = torch.cat([total_loss.clone().detach().view(1), total_tokens.view(1)])
    
    return (total_loss, total_tokens, {'lm loss': (reporting_loss)})


def forward_step(data_iterator, model):
    """Forward step for MIMO model training.
    
    Args:
        data_iterator: iterator over the dataset
        model: MIMO model instance
    
    Returns:
        tuple: (output_tensor, loss_function)
    """
    data_batch = get_batch(data_iterator)
    if data_batch is None:
        data_batch = {'input_ids': None}
    output_tensor, loss_mask = model(**data_batch)
    return output_tensor, partial(loss_func, loss_mask)


def train_homogeneous_parallelism(
    model_config: ModelConfig,
    data_config: DataConfig,
    runtime_config: RuntimeConfig,
):
    """Train VLM with homogeneous parallelism using no-pipelining schedule.
    
    This function is optimized for Case 1 where:
    - Both vision and LLM use the same parallelism strategy
    - PP=1 for all modules (no pipeline parallelism)
    - Uses forward_backward_no_pipelining schedule
    - Both modules share the same grid and process groups
    
    Args:
        model_config: Model configuration (architectures, parallelisms)
        data_config: Data configuration (batch sizes, dataset params)
        runtime_config: Runtime configuration (iterations, profiling, etc)
    """
    logging.info("Creating VLM MIMO model with homogeneous parallelism...")
    
    # Create MIMO model with shared grid and pg_collection
    # This function validates that both modules have identical parallelism settings
    mimo_model, shared_grid, shared_pg_collection, topology = get_vlm_mimo_model_homogeneous(
        model_config=model_config,
        seq_len=data_config.seq_length,
    )
    
    # Disable barrier for multimodule setup
    mimo_model.config.barrier_with_L1_time = False
    
    logging.info(f"Rank {dist.get_rank()}: Model created successfully")
    
    # For homogeneous parallelism, create a simple module-to-grid tuple
    # Both modules use the same shared grid
    module_to_grid_tuple = [
        (mimo_model.modality_submodules[model_config.encoder_module_name], shared_grid),
        (mimo_model.language_model, shared_grid)
    ]
    
    # Configure gradient synchronization functions
    mimo_model.config.no_sync_func = partial(multimodule_no_sync, module_to_grid_tuple=module_to_grid_tuple)
    mimo_model.config.finalize_model_grads_func = partial(finalize_model_grads, module_to_grid_tuple=module_to_grid_tuple)
    
    logging.info(f"Rank {dist.get_rank()}: Creating data iterator...")
    
    # Get data iterator - create a simple module_to_grid_map for data iterator
    module_to_grid_map = {
        model_config.encoder_module_name: shared_grid,
        model_config.llm_module_name: shared_grid
    }
    data_iterator = get_data_iterator(
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
    
    # Set model type
    mimo_model.model_type = 'unit-test'

    # Initialize PyTorch profiler if requested
    prof = None
    if runtime_config.enable_profiling and runtime_config.use_pytorch_profiler:
        prof = torch.profiler.profile(
            schedule=torch.profiler.schedule(
                wait=max(runtime_config.profile_start_step - 1, 0) if runtime_config.profile_start_step else 0,
                warmup=1 if runtime_config.profile_start_step and runtime_config.profile_start_step > 0 else 0,
                active=(runtime_config.profile_end_step - runtime_config.profile_start_step) if runtime_config.profile_start_step and runtime_config.profile_end_step else runtime_config.num_iterations,
                repeat=1,
            ),
            on_trace_ready=torch.profiler.tensorboard_trace_handler(runtime_config.tensorboard_dir) if runtime_config.tensorboard_dir else None,
            record_shapes=True,
            with_stack=True,
        )
        prof.start()
    
    all_losses = []
    
    
    for iteration in range(runtime_config.num_iterations):
        # Start iteration timing
        perf_monitor.start_iteration()
        
        # Handle profiling
        if runtime_config.enable_profiling:
            if runtime_config.use_pytorch_profiler:
                if prof:
                    prof.step()
            elif runtime_config.profile_start_step is not None and iteration == runtime_config.profile_start_step:
                logging.info(f"Rank {dist.get_rank()}: Starting profiler at iteration {iteration}")
                torch.cuda.cudart().cudaProfilerStart()
        
        # Run forward_backward_no_pipelining schedule
        # Note: This schedule doesn't use P2P communicator since there's no pipeline parallelism
        losses_reduced = schedule.forward_backward_no_pipelining(
            forward_step_func=forward_step,
            data_iterator=data_iterator,
            model=[mimo_model],
            num_microbatches=data_config.num_microbatches,
            seq_length=data_config.seq_length,
            micro_batch_size=data_config.base_batch_size,
            forward_only=False,
            pg_collection=shared_pg_collection,
        )
        
        all_losses.append(losses_reduced)
        
        # End iteration timing
        perf_monitor.end_iteration()
        
        # Log performance metrics
        perf_monitor.log_performance(
            iteration=iteration + 1,
        )
        
        # Zero gradients for next iteration
        zero_grad_buffer_for_multimodule(module_to_grid_tuple)
        
        # Stop CUDA profiling if enabled
        if runtime_config.enable_profiling and not runtime_config.use_pytorch_profiler:
            if runtime_config.profile_end_step is not None and iteration == runtime_config.profile_end_step:
                logging.info(f"Rank {dist.get_rank()}: Stopping profiler at iteration {iteration}")
                torch.cuda.cudart().cudaProfilerStop()
    
    # Stop PyTorch profiler if running
    if prof:
        prof.stop()
    
    # Save metrics
    metrics_dir = runtime_config.metrics_output_dir
    os.makedirs(metrics_dir, exist_ok=True)
    
    vision_arch = model_config.get_arch(model_config.encoder_module_name)
    llm_arch = model_config.get_arch(model_config.llm_module_name)
    
    metrics_file = os.path.join(
        metrics_dir, 
        f"metrics_vl{vision_arch.num_layers}_ll{llm_arch.num_layers}_mb{data_config.num_microbatches}.json"
    )
    
    perf_monitor.save_metrics_to_file(
        filepath=metrics_file,
        extra_info={
            'exp_name': 'homogeneous_parallelism',
            'schedule': 'forward_backward_no_pipelining',
        },
        exclude_warmup=True
    )
    
    logging.info(f"Rank {dist.get_rank()}: Training completed.")
    
    return all_losses


if __name__ == "__main__":
    # Initialize distributed training
    Utils.initialize_distributed()
    
    # Create centralized configurations
    model_config = ModelConfig(
        module_architectures={
            'images': ModuleArchConfig(
                num_layers=40,
                hidden_size=1408,
                num_attention_heads=16,
                seq_length=1024,
                vocab_size=0,  # Vision encoder has no vocab
            ),
            'language_module': ModuleArchConfig(
                num_layers=80,
                hidden_size=8192,
                num_attention_heads=64,
                seq_length=4096,
                vocab_size=32000,
            ),
        },
        module_parallelisms={
            'images': ModuleParallelismConfig(
                tensor_parallel=8,
                pipeline_parallel=1,
                data_parallel=1,
            ),
            'language_module': ModuleParallelismConfig(
                tensor_parallel=8,
                pipeline_parallel=1,
                data_parallel=1,
            ),
        },
        special_token_ids={'images': 32000},
        llm_module_name='language_module',
    )
    
    data_config = DataConfig(
        base_batch_size=1,
        num_microbatches=16,
        seq_length=4096,
        image_seq_length=1024,
        vocab_size=32000,
        image_special_token_id=32000,
        dataset_size=4096,
        num_workers=8,
        prefetch_factor=4,
    )
    
    runtime_config = RuntimeConfig(
        num_iterations=10,
        warmup_iterations=2,
        log_interval=1,
        enable_performance_monitoring=True,
        metrics_output_dir=os.environ.get("METRICS_OUTPUT_DIR", "./metrics"),
        enable_profiling=False,
        use_pytorch_profiler=os.environ.get("USE_PYTORCH_PROFILER", "True").lower() == "true",
        profile_start_step=3,
        profile_end_step=8,
        tensorboard_dir=os.environ.get("PROFILE_OUTPUT_DIR", "./tb_logs"),
    )
    
    # Run training
    losses = train_homogeneous_parallelism(
        model_config=model_config,
        data_config=data_config,
        runtime_config=runtime_config,
    )
    
    dist.destroy_process_group()

