"""
CUDA_VISIBLE_DEVICES=0,1 USE_PYTORCH_PROFILER=False nsys profile -w true -t cublas,cuda,nvtx,osrt -s cpu -c cudaProfilerApi -o profile_output uv run python -m torch.distributed.run --nproc_per_node=2 tests/unit_tests/models/heterogenous_parallel/train.py
uv run python -m torch.distributed.run --nproc_per_node=8 tests/unit_tests/models/heterogenous_parallel/train.py
"""

import torch
import torch.distributed as dist
from functools import partial
import logging


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', force=True)

from tests.unit_tests.test_utilities import Utils
from tests.unit_tests.models.heterogenous_parallel.model_specs import get_vlm_mimo_model
from tests.unit_tests.models.heterogenous_parallel.parallel_utils import (
    get_module_to_grid_tuple, 
    multimodule_no_sync, 
    finalize_model_grads,
    get_pg_collections_for_rank,
    zero_grad_buffer_for_multimodule,
    is_current_rank_in_grid,
)
from tests.unit_tests.models.heterogenous_parallel.dp_aware_data_iterator import get_data_iterator, get_batch
from tests.unit_tests.models.heterogenous_parallel.performance_utils import create_performance_monitor
from tests.unit_tests.models.heterogenous_parallel.config import (
    ModelConfig, ModuleArchConfig, ModuleParallelismConfig,
    DataConfig, RuntimeConfig
)
from megatron.core.pipeline_parallel.multimodule_communicator import MultiModulePipelineCommunicator
import megatron.core.pipeline_parallel.schedules as schedule
import os

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
    # Return output and loss function
    return output_tensor, partial(loss_func, loss_mask)


def test_1f_1b_schedule_vlm_mimo_model_custom_pgs(
    model_config: ModelConfig,
    data_config: DataConfig,
    runtime_config: RuntimeConfig,
):
    """Test 1F1B schedule with VLM MIMO model using centralized configs.
    
    Args:
        model_config: Model configuration (architectures, parallelisms)
        data_config: Data configuration (batch sizes, dataset params)
        runtime_config: Runtime configuration (iterations, profiling, etc)
    """
    logging.info("Creating VLM MIMO model...")
    
    # Create MIMO model using simplified interface
    mimo_model, module_to_grid_map, topology = get_vlm_mimo_model(
        model_config=model_config,
        seq_len=data_config.seq_length,
    )
    

    mimo_model.config.barrier_with_L1_time = False
    
    logging.info(f"Rank {dist.get_rank()}: Model created successfully")
    
    # Set up module to grid tuple for no_sync and finalize_model_grads
    module_to_grid_tuple = get_module_to_grid_tuple(
        mimo_model, 
        module_to_grid_map[model_config.encoder_module_name], 
        module_to_grid_map[model_config.llm_module_name]
    )
    
    # Configure no_sync and finalize_model_grads functions
    mimo_model.config.no_sync_func = partial(multimodule_no_sync, module_to_grid_tuple=module_to_grid_tuple)
    mimo_model.config.finalize_model_grads_func = partial(finalize_model_grads, module_to_grid_tuple=module_to_grid_tuple)
    
    logging.info(f"Rank {dist.get_rank()}: Creating data iterator...")
    
    # Get data iterator using simplified interface
    data_iterator = get_data_iterator(
        model_config=model_config,
        data_config=data_config,
        module_to_grid_map=module_to_grid_map,
    )
    
    # Create performance monitor using simplified interface
    perf_monitor = create_performance_monitor(
        model_config=model_config,
        data_config=data_config,
        runtime_config=runtime_config,
        megatron_config=mimo_model.config,
    )
    
    logging.info(f"Rank {dist.get_rank()}: Performance monitor initialized")
    
    # Create multimodule communicator
    multimodule_communicator = MultiModulePipelineCommunicator(
        module_to_grid_map, topology, mimo_model.config, dim_mapping={'b': 0, 's': 1, 'h': 2}
    )
    
    # Set model type for unit test
    mimo_model.model_type = 'unit-test'
    
    # Prepare common arguments for schedule
    common_args = {
        'forward_step_func': forward_step,
        'data_iterator': data_iterator,
        'model': [mimo_model],
        'num_microbatches': data_config.num_microbatches,
        'seq_length': data_config.seq_length,
        'micro_batch_size': data_config.base_batch_size,
        'forward_only': False,
    }
    
    # Get pg collections for modules that should be initialized on this rank
    pg_collection = get_pg_collections_for_rank(module_to_grid_map)
    
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
        
        # Run 1F1B schedule
        losses_reduced = schedule.forward_backward_pipelining_without_interleaving(
            p2p_communicator=multimodule_communicator, 
            pg_collection=pg_collection, 
            **common_args
        )

        
        all_losses.append(losses_reduced)
        
        # End iteration timing
        perf_monitor.end_iteration()

        # Log performance metrics
        perf_monitor.log_performance(
            iteration=iteration + 1,
        )
        
        zero_grad_buffer_for_multimodule(module_to_grid_tuple)
        
        # Stop CUDA profiling if enabled
        if runtime_config.enable_profiling and not runtime_config.use_pytorch_profiler:
            if runtime_config.profile_end_step is not None and iteration == runtime_config.profile_end_step:
                logging.info(f"Rank {dist.get_rank()}: Stopping profiler at iteration {iteration}")
                torch.cuda.cudart().cudaProfilerStop()
    
    # Stop PyTorch profiler if running
    if prof:
        prof.stop()
    

    
    metrics_dir = runtime_config.metrics_output_dir
    os.makedirs(metrics_dir, exist_ok=True)
    
    vision_arch = model_config.get_arch('images')
    llm_arch = model_config.get_arch(model_config.llm_module_name)
    
    metrics_file = os.path.join(
        metrics_dir, 
        f"metrics_vl{vision_arch.num_layers}_ll{llm_arch.num_layers}_mb{data_config.num_microbatches}.json"
    )
    perf_monitor.save_metrics_to_file(
        filepath=metrics_file,
        extra_info={
            'exp_name': 'vlm_mimo_model_custom_pgs',
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
                num_layers=4,
                hidden_size=768,
                num_attention_heads=4,
                seq_length=256,
                vocab_size=0,  # Vision encoder has no vocab
            ),
            'language_module': ModuleArchConfig(
                num_layers=4,
                hidden_size=768,
                num_attention_heads=4,
                seq_length=1024,
                vocab_size=4000,
            ),
        },
        module_parallelisms={
            'images': ModuleParallelismConfig(
                tensor_parallel=1,
                pipeline_parallel=1,
                data_parallel=1,
            ),
            'language_module': ModuleParallelismConfig(
                tensor_parallel=1,
                pipeline_parallel=1,
                data_parallel=1,
            ),
        },
        special_token_ids={'images': 32000},
        llm_module_name='language_module',
    )
    
    data_config = DataConfig(
        base_batch_size=8,
        num_microbatches=16,
        seq_length=2048,
        image_seq_length=1024,
        vocab_size=4000,
        image_special_token_id=32000,
        dataset_size=4096,
        num_workers=8,
        prefetch_factor=4,
    )
    
    runtime_config = RuntimeConfig(
        num_iterations=6,
        warmup_iterations=2,
        log_interval=1,
        enable_performance_monitoring=True,
        metrics_output_dir=os.environ.get("METRICS_OUTPUT_DIR", "./metrics"),
        enable_profiling=False,
        use_pytorch_profiler=os.environ.get("USE_PYTORCH_PROFILER", "True").lower() == "true",
        profile_start_step=3,
        profile_end_step=5,
        tensorboard_dir=os.environ.get("PROFILE_OUTPUT_DIR", "./tb_logs"),
    )
    
    # Run training
    losses = test_1f_1b_schedule_vlm_mimo_model_custom_pgs(
        model_config=model_config,
        data_config=data_config,
        runtime_config=runtime_config,
    )

    dist.destroy_process_group()