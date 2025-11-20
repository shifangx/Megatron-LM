import torch
import torch.distributed as dist
from functools import partial
import logging
import os
from typing import Dict, List
from torch.utils.data import DataLoader, Dataset

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', force=True)

from tests.unit_tests.test_utilities import Utils
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_with_transformer_engine_spec
from megatron.core.models.gpt.gpt_model import GPTModel
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.distributed import DistributedDataParallel, DistributedDataParallelConfig
from megatron.core.pipeline_parallel.p2p_communication import P2PCommunicator
from megatron.core.hyper_comm_grid import HyperCommGrid
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.distributed.finalize_model_grads import finalize_model_grads
import megatron.core.pipeline_parallel.schedules as schedule


def get_gpt_model(
    num_layers, hidden_size, vocab_size, seq_length, 
    tp_size, pp_size, dp_size, seed=42
):
    """Create a GPT model with the specified parallelism configuration.
    
    Args:
        num_layers: Number of transformer layers
        hidden_size: Hidden size
        vocab_size: Vocabulary size
        seq_length: Sequence length
        tp_size: Tensor parallel size
        pp_size: Pipeline parallel size
        dp_size: Data parallel size
        seed: Random seed
    
    Returns:
        Tuple of (model, transformer_config)
    """
    # Initialize model parallel first
    Utils.initialize_model_parallel(
        tensor_model_parallel_size=tp_size,
        pipeline_model_parallel_size=pp_size,
        context_parallel_size=1,
    )
    
    # Then set seeds
    torch.manual_seed(seed)
    model_parallel_cuda_manual_seed(seed)
    
    from megatron.core import parallel_state as ps
    
    # Check if this rank participates in the model
    pp_rank = ps.get_pipeline_model_parallel_rank()
    pp_world_size = ps.get_pipeline_model_parallel_world_size()
    pre_process = ps.is_pipeline_first_stage()
    post_process = ps.is_pipeline_last_stage()
    
    logging.info(f"Rank {dist.get_rank()}: PP rank {pp_rank}/{pp_world_size}, "
                 f"pre_process={pre_process}, post_process={post_process}")
    
    # Create transformer config
    transformer_config = TransformerConfig(
        num_layers=num_layers,
        hidden_size=hidden_size,
        num_attention_heads=8,
        use_cpu_initialization=True,
        pipeline_dtype=torch.bfloat16,
        bf16=True,
        tensor_model_parallel_size=tp_size,
        pipeline_model_parallel_size=pp_size,
    )
    
    # Get layer spec
    layer_spec = get_gpt_layer_with_transformer_engine_spec()
    
    # Create GPT model
    model = GPTModel(
        config=transformer_config,
        transformer_layer_spec=layer_spec,
        vocab_size=vocab_size,
        max_sequence_length=seq_length,
        pre_process=pre_process,
        post_process=post_process,
    )
    
    # Move model to GPU and convert to bfloat16
    model.to(torch.device("cuda")).to(torch.bfloat16)
    
    # Wrap with DDP if needed
    if dp_size > 1:
        ddp_config = DistributedDataParallelConfig(
            overlap_grad_reduce=True, 
            bucket_size=10000
        )
        model = DistributedDataParallel(
            config=transformer_config,
            ddp_config=ddp_config,
            module=model,
        )
    
    logging.info(f"Rank {dist.get_rank()}: GPT model created successfully")
    
    return model, transformer_config


class GPTMockDataset(Dataset):
    """Mock dataset for GPT training with better profiling performance.
    
    Generates data on CPU for better dataloader performance with workers.
    """
    
    def __init__(self, size: int, seq_length: int, vocab_size: int):
        """
        Args:
            size: Dataset size (number of samples)
            seq_length: Sequence length for tokens
            vocab_size: Vocabulary size
        """
        self.size = size
        self.seq_length = seq_length
        self.vocab_size = vocab_size
    
    def __len__(self):
        return self.size
    
    def __getitem__(self, idx):
        """Generate a single sample on CPU."""
        # Generate random tokens and labels on CPU
        tokens = torch.randint(0, self.vocab_size, (self.seq_length,), dtype=torch.long)
        labels = torch.randint(0, self.vocab_size, (self.seq_length,), dtype=torch.long)
        
        return {'tokens': tokens, 'labels': labels}


def _collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """
    Collate function for the DataLoader.
    
    Args:
        batch: List of dictionaries from the dataset
    
    Returns:
        Dictionary of batched tensors
    """
    tokens = torch.stack([item['tokens'] for item in batch])
    labels = torch.stack([item['labels'] for item in batch])
    
    return {'tokens': tokens, 'labels': labels}


def move_to_device(data, device):
    """
    Recursively move tensors to device with non_blocking for async transfers.
    
    When pin_memory=True in DataLoader, non_blocking=True enables async GPU transfers.
    This allows the GPU to continue processing while the next batch is being transferred.
    """
    if isinstance(data, torch.Tensor):
        return data.to(device, non_blocking=True)
    elif isinstance(data, dict):
        return {k: move_to_device(v, device) for k, v in data.items()}
    elif isinstance(data, list):
        return [move_to_device(v, device) for v in data]
    return data


def get_gpt_data_iterator(batch_size, seq_length, vocab_size, num_workers=8, prefetch_factor=4):
    """Create an optimized data iterator for GPT training.
    
    Args:
        batch_size: Batch size
        seq_length: Sequence length
        vocab_size: Vocabulary size
        num_workers: Number of dataloader workers (default: 8)
        prefetch_factor: Number of batches to prefetch per worker (default: 4)
    
    Returns:
        Iterator that yields batches of data
    """
    dataset = GPTMockDataset(
        size=256,  # Dataset size - can be adjusted as needed
        seq_length=seq_length,
        vocab_size=vocab_size
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=_collate_fn,
        pin_memory=True,  # CRITICAL: Enables fast CPU->GPU transfers
        persistent_workers=True,  # Keep workers alive between epochs - reduces overhead
        prefetch_factor=prefetch_factor,  # Prefetch batches per worker
    )
    
    return iter(dataloader)


def loss_func(loss_mask, output_tensor):
    """Simple loss function for GPT training.
    
    Args:
        loss_mask: Mask indicating which tokens contribute to the loss
        output_tensor: Model output tensor
    
    Returns:
        tuple: (loss, num_tokens, metrics_dict)
    """
    losses = output_tensor.float()
    
    loss_mask = loss_mask.contiguous().view(-1).float()
    
    total_tokens = loss_mask.sum().clone().detach().to(torch.int)
    total_loss = torch.sum(losses.view(-1) * loss_mask)
    reporting_loss = torch.cat([total_loss.clone().detach().view(1), total_tokens.view(1)])
    
    return (total_loss, total_tokens, {'lm loss': reporting_loss})


def forward_step(data_iterator, model):
    """Forward step for GPT training.
    
    Args:
        data_iterator: Iterator over the dataset
        model: GPT model instance
    
    Returns:
        tuple: (output_tensor, loss_function)
    """
    from megatron.core import parallel_state as ps
    
    # Get data batch and move to GPU asynchronously
    data = next(data_iterator)
    data = move_to_device(data, torch.device("cuda"))
    
    tokens = data['tokens']
    labels = data['labels']
    
    # Create attention mask (all ones for simplicity)
    attention_mask = torch.ones(
        tokens.shape[0], 1, tokens.shape[1], tokens.shape[1],
        dtype=torch.bool, device=tokens.device
    )
    
    # Create position IDs
    position_ids = torch.arange(tokens.shape[1], dtype=torch.long, device=tokens.device)
    position_ids = position_ids.unsqueeze(0).expand_as(tokens)
    
    # Create loss mask (all ones for simplicity)
    loss_mask = torch.ones(tokens.shape, dtype=torch.float, device=tokens.device)
    
    # Forward pass
    if ps.is_pipeline_last_stage():
        output_tensor = model(
            input_ids=tokens,
            position_ids=position_ids,
            attention_mask=attention_mask,
            labels=labels,
        )
    else:
        output_tensor = model(
            input_ids=tokens,
            position_ids=position_ids,
            attention_mask=attention_mask,
        )
    
    # Return output and loss function
    return output_tensor, partial(loss_func, loss_mask)


def train_gpt_model(
    num_layers, hidden_size, vocab_size, seq_length,
    tp_size, pp_size, dp_size,
    batch_size, num_microbatches,
    num_iterations=1, 
    profile_start_step=None, 
    profile_end_step=None, 
    enable_profiling=False,
    use_pytorch_profiler=False, 
    tensorboard_dir=None,
    seed=42
):
    """Train GPT model with profiling support.
    
    Args:
        num_layers: Number of transformer layers
        hidden_size: Hidden size
        vocab_size: Vocabulary size
        seq_length: Sequence length
        tp_size: Tensor parallel size
        pp_size: Pipeline parallel size
        dp_size: Data parallel size
        batch_size: Batch size
        num_microbatches: Number of microbatches for pipeline parallelism
        num_iterations: Number of training iterations
        profile_start_step: Step to start profiling (None = don't profile)
        profile_end_step: Step to end profiling
        enable_profiling: Whether to enable profiling
        use_pytorch_profiler: Use PyTorch profiler (True) or CUDA profiler (False)
        tensorboard_dir: TensorBoard output directory (for PyTorch profiler)
        seed: Random seed
    
    Returns:
        List of losses from all iterations
    """
    logging.info(f"Creating GPT model with {num_layers} layers, hidden size {hidden_size}...")
    
    # Create model
    model, transformer_config = get_gpt_model(
        num_layers=num_layers,
        hidden_size=hidden_size,
        vocab_size=vocab_size,
        seq_length=seq_length,
        tp_size=tp_size,
        pp_size=pp_size,
        dp_size=dp_size,
        seed=seed,
    )
    
    # Create optimized data iterator with persistent workers and pin memory
    logging.info(f"Creating optimized data iterator with persistent workers...")
    data_iterator = get_gpt_data_iterator(
        batch_size=batch_size,
        seq_length=seq_length,
        vocab_size=vocab_size,
        num_workers=8,  # Use multiple workers for better CPU utilization
        prefetch_factor=4,  # Prefetch 4 batches per worker
    )
    
    # Set model type for schedule
    model.model_type = 'unit-test'
    
    # Prepare common arguments for schedule
    common_args = {
        'forward_step_func': forward_step,
        'data_iterator': data_iterator,
        'model': [model],
        'num_microbatches': num_microbatches,
        'seq_length': seq_length,
        'micro_batch_size': batch_size,
        'forward_only': False,
    }
    
    # Initialize PyTorch profiler if requested
    prof = None
    if enable_profiling and use_pytorch_profiler:
        prof = torch.profiler.profile(
            schedule=torch.profiler.schedule(
                wait=max(profile_start_step - 1, 0) if profile_start_step else 0,
                warmup=1 if profile_start_step and profile_start_step > 0 else 0,
                active=(profile_end_step - profile_start_step) if profile_start_step and profile_end_step else num_iterations,
                repeat=1,
            ),
            on_trace_ready=torch.profiler.tensorboard_trace_handler(tensorboard_dir) if tensorboard_dir else None,
            record_shapes=True,
            with_stack=True,
        )
        prof.start()
    
    all_losses = []
    
    for iteration in range(num_iterations):
        # Handle profiling
        if enable_profiling:
            if use_pytorch_profiler:
                if prof:
                    prof.step()
            elif profile_start_step is not None and iteration == profile_start_step:
                logging.info(f"Rank {dist.get_rank()}: Starting CUDA profiler at iteration {iteration}")
                torch.cuda.cudart().cudaProfilerStart()
        
        logging.info(f"Rank {dist.get_rank()}: Iteration {iteration} - Starting training...")
        
        # Run forward-backward pass
        if pp_size == 1:
            # No pipeline parallelism
            losses_reduced = schedule.forward_backward_no_pipelining(
                **common_args
            )
        else:
            # With pipeline parallelism
            losses_reduced = schedule.forward_backward_pipelining_without_interleaving(
                **common_args
            )
        
        all_losses.append(losses_reduced)
        logging.info(f"Rank {dist.get_rank()}: Iteration {iteration} - Losses: {losses_reduced}")
        
        # Finalize gradients - only if using DDP wrapper
        if isinstance(model, DistributedDataParallel):
            if hasattr(model, 'finish_grad_sync'):
                model.finish_grad_sync()
        
        # Zero gradients for next iteration
        if isinstance(model, DistributedDataParallel):
            for param in model.module.parameters():
                if param.grad is not None:
                    param.grad.zero_()
        else:
            for param in model.parameters():
                if param.grad is not None:
                    param.grad.zero_()
        
        # Stop CUDA profiling if enabled
        if enable_profiling and not use_pytorch_profiler:
            if profile_end_step is not None and iteration == profile_end_step:
                logging.info(f"Rank {dist.get_rank()}: Stopping CUDA profiler at iteration {iteration}")
                torch.cuda.cudart().cudaProfilerStop()
    
    # Stop PyTorch profiler if running
    if prof:
        prof.stop()
    
    logging.info(f"Rank {dist.get_rank()}: Training completed. All losses: {all_losses}")
    
    return all_losses


if __name__ == "__main__":
    # Initialize distributed training
    Utils.initialize_distributed()
    
    # Profiling configuration - read from environment variables
    enable_profiling = True
    use_pytorch_profiler = os.environ.get("USE_PYTORCH_PROFILER", "True").lower() == "true"
    tensorboard_dir = os.environ.get("PROFILE_OUTPUT_DIR", "./gpt_tb_logs")
    num_iterations = 6
    profile_start_step = 3
    profile_end_step = 5
    
    # Model parameters
    num_layers = 16
    hidden_size = 2048
    
    # Data parameters
    vocab_size = 48000
    seq_length = 4096
    
    # Model parallelism
    tp_size = int(os.environ.get("TP_SIZE", "1"))
    pp_size = int(os.environ.get("PP_SIZE", "2"))
    dp_size = int(os.environ.get("DP_SIZE", "1"))
    
    # Training parameters
    batch_size = 2
    num_microbatches = 16
    
    logging.info(f"Starting GPT training with TP={tp_size}, PP={pp_size}, DP={dp_size}")
    
    losses = train_gpt_model(
        num_layers=num_layers,
        hidden_size=hidden_size,
        vocab_size=vocab_size,
        seq_length=seq_length,
        tp_size=tp_size,
        pp_size=pp_size,
        dp_size=dp_size,
        batch_size=batch_size,
        num_microbatches=num_microbatches,
        num_iterations=num_iterations,
        profile_start_step=profile_start_step,
        profile_end_step=profile_end_step,
        enable_profiling=enable_profiling,
        use_pytorch_profiler=use_pytorch_profiler,
        tensorboard_dir=tensorboard_dir,
    )
    
    logging.info(f"Final losses: {losses}")
    
    # Cleanup
    Utils.destroy_model_parallel()
    dist.destroy_process_group()
