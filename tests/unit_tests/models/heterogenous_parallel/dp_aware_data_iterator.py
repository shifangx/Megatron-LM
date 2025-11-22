"""
Hybrid DP-Aware Data Iterator for Heterogeneous Multi-Module Training

Simplified implementation that works directly with grids to create
DP-aware data loaders for MIMO models.
"""

from typing import Dict, Optional, Any, Iterator, List
import logging
import torch
import torch.distributed as dist
from torch.utils.data import Sampler, DataLoader
from examples.mimo.data.mock import MockVLMDataset
from megatron.core.pipeline_parallel.utils import is_pp_first_stage, is_pp_last_stage
from tests.unit_tests.models.heterogenous_parallel.config import DataConfig, ModelConfig




def validate_heterogeneous_batch_config(
    model_config: ModelConfig,
    data_config: DataConfig,
) -> int:
    """Validate that batch configuration is compatible with DP sizes across modules.
    
    Args:
        model_config: Model configuration with parallelism settings
        data_config: Data configuration with base_batch_size
        
    Returns:
        global_batch_size: The validated global batch size
        
    Raises:
        ValueError: If batch configuration is invalid for any module's DP size
    """
    # Get LLM DP size and calculate global batch
    llm_parallelism = model_config.get_parallelism(model_config.llm_module_name)
    llm_dp_size = llm_parallelism.data_parallel
    global_batch_size = data_config.base_batch_size * llm_dp_size
    
    # Validate all modules can split the global batch evenly
    for module_name in model_config.module_parallelisms.keys():
        module_parallelism = model_config.get_parallelism(module_name)
        dp_size = module_parallelism.data_parallel
        
        if global_batch_size % dp_size != 0:
            raise ValueError(
                f"Invalid DP configuration for module '{module_name}': "
                f"global_batch_size ({global_batch_size}) must be divisible by "
                f"dp_size ({dp_size}). "
                f"LLM: base_batch_size={data_config.base_batch_size}, dp_size={llm_dp_size}. "
                f"Consider adjusting base_batch_size or DP configurations."
            )
        
        micro_batch_size = global_batch_size // dp_size
        
        if dist.get_rank() == 0:
            logging.info(
                f"Module '{module_name}': DP={dp_size}, "
                f"micro_batch_size={micro_batch_size} (per DP replica)"
            )
    
    if dist.get_rank() == 0:
        logging.info(
            f"Validated batch config: base_batch={data_config.base_batch_size}, "
            f"global_batch={global_batch_size}, num_microbatches={data_config.num_microbatches}"
        )
    
    return global_batch_size


class HeterogeneousDPBatchSampler(Sampler):
    """DP-aware batch sampler for heterogeneous multi-module training.
    
    Ensures different DP replicas within the same module receive non-overlapping
    data while respecting module-specific batch sizes.
    """
    
    def __init__(
        self,
        dataset_size: int,
        global_batch_size: int,
        dp_rank: int,
        dp_size: int,
        drop_last: bool = True
    ):
        """
        Args:
            dataset_size: Total number of samples in the dataset
            global_batch_size: Global batch size (same across all modules)
            dp_rank: Data parallel rank within the module
            dp_size: Data parallel size for the module
            drop_last: Whether to drop the last incomplete batch
        """
        self.dataset_size = dataset_size
        self.global_batch_size = global_batch_size
        self.dp_rank = dp_rank
        self.dp_size = dp_size
        self.drop_last = drop_last
        
        # Calculate micro-batch size for this DP replica
        self.micro_batch_size = global_batch_size // dp_size
        
        # Calculate number of complete batches
        self.num_batches = dataset_size // global_batch_size
        if not drop_last and dataset_size % global_batch_size != 0:
            self.num_batches += 1
    
    def __iter__(self):
        """Generate batches for this DP rank."""
        for batch_idx in range(self.num_batches):
            # Calculate global indices for this microbatch
            start_idx = batch_idx * self.global_batch_size
            end_idx = min(start_idx + self.global_batch_size, self.dataset_size)
            
            # Shard for this DP rank
            my_indices = list(range(
                start_idx + self.dp_rank,
                end_idx,
                self.dp_size
            ))
            
            # Handle incomplete last batch
            if len(my_indices) == self.micro_batch_size:
                yield my_indices
            elif not self.drop_last and len(my_indices) > 0:
                yield my_indices
    
    def __len__(self):
        """Number of batches this sampler will yield."""
        return self.num_batches


def _collate_fn(
    batch: List[Dict], 
    image_seq_length: int = 1024, 
    hidden_size: int = 1024
) -> Dict[str, torch.Tensor]:
    """Collate function for the DataLoader."""
    input_ids = torch.stack([item["input_ids"] for item in batch])
    labels = torch.stack([item["labels"] for item in batch])
    loss_mask = torch.stack([item["loss_mask"] for item in batch])
    position_ids = torch.stack([item["position_ids"] for item in batch])
    
    bsz = input_ids.shape[0]
    hidden_states = torch.zeros(image_seq_length, bsz, hidden_size, dtype=torch.bfloat16)
    
    return {
        "input_ids": input_ids,
        "labels": labels,
        "loss_mask": loss_mask,
        "position_ids": position_ids,
        "modality_inputs": {
            "images": {
                "clip_encoder": {'hidden_states': hidden_states, 'attention_mask': None},
            }
        },
    }


def move_to_device(data, device):
    """Recursively move tensors in nested dicts to device with non_blocking for async transfers."""
    if isinstance(data, torch.Tensor):
        return data.to(device, non_blocking=True)
    elif isinstance(data, dict):
        return {k: move_to_device(v, device) for k, v in data.items()}
    elif isinstance(data, list):
        return [move_to_device(v, device) for v in data]
    return data


def get_data_iterator(
    model_config: ModelConfig,
    data_config: DataConfig,
    module_to_grid_map: Dict[str, Any],
) -> Optional[Iterator]:
    """Create DP-aware data iterator for heterogeneous multi-module training.
    
    Args:
        model_config: Model configuration with parallelism settings
        data_config: Data configuration (batch sizes, dataset params)
        module_to_grid_map: Maps module names to their HyperCommGrids
        
    Returns:
        Data iterator for this rank's module, or None if rank not involved
        
    Raises:
        ValueError: If batch configuration is incompatible with DP sizes
    """
    # Validate batch configuration and get global batch size
    global_batch_size = validate_heterogeneous_batch_config(
        model_config=model_config,
        data_config=data_config,
    )
    
    # Find current rank's grid and get DP info
    current_rank = dist.get_rank()
    my_grid = None
    for grid in module_to_grid_map.values():
        if grid.rank_offset <= current_rank < (grid.rank_offset + grid.size):
            my_grid = grid
            break
    
    if my_grid is None:
        return None
    
    # Get DP info from current rank's grid
    dp_size, dp_rank = my_grid.get_pg("dp").size(), my_grid.get_pg("dp").rank()
    
    # Check if we should initialize iterator on this rank
    encoder_grid = module_to_grid_map.get(model_config.encoder_module_name)
    llm_grid = module_to_grid_map.get(model_config.llm_module_name)
    
    def is_rank_in_grid(grid):
        if grid is None:
            return False
        return grid.rank_offset <= current_rank < (grid.rank_offset + grid.size)
    
    encoder_condition = (
        encoder_grid is not None and
        is_rank_in_grid(encoder_grid) and 
        is_pp_first_stage(encoder_grid.get_pg("pp"))
    )
    
    llm_condition = (
        llm_grid is not None and
        is_rank_in_grid(llm_grid) and 
        (is_pp_first_stage(llm_grid.get_pg("pp")) or is_pp_last_stage(llm_grid.get_pg("pp")))
    )
    
    if not (encoder_condition or llm_condition):
        return None
    
    # Create dataset
    dataset = MockVLMDataset(
        size=data_config.dataset_size,
        image_size=224,
        seq_len=data_config.seq_length,
        image_seq_length=data_config.image_seq_length,
        pad_token_id=0,
        image_token_id=data_config.image_special_token_id
    )
    
    # Create DP-aware batch sampler
    batch_sampler = HeterogeneousDPBatchSampler(
        dataset_size=len(dataset),
        global_batch_size=global_batch_size,
        dp_rank=dp_rank,
        dp_size=dp_size,
        drop_last=True
    )
    
    # Get vision hidden size from model config
    vision_hidden_size = model_config.get_arch(model_config.encoder_module_name).hidden_size
    
    # Create DataLoader
    dataloader = DataLoader(
        dataset,
        batch_sampler=batch_sampler,
        num_workers=data_config.num_workers,
        collate_fn=lambda batch: _collate_fn(
            batch, 
            image_seq_length=data_config.image_seq_length, 
            hidden_size=vision_hidden_size
        ),
        pin_memory=data_config.pin_memory,
        persistent_workers=data_config.persistent_workers,
        prefetch_factor=data_config.prefetch_factor,
    )
    
    return iter(dataloader)


def get_batch(data_iterator: Optional[Iterator]) -> Optional[Dict[str, Any]]:
    """Get next batch from data iterator and move to GPU.
    
    Args:
        data_iterator: Data iterator to get batch from
        
    Returns:
        Batch moved to CUDA device, or None if iterator is None
    """
    if data_iterator is not None:
        input_tensor = next(data_iterator)
        if input_tensor is not None:
            input_tensor = move_to_device(input_tensor, torch.device("cuda"))
        return input_tensor
    return None
