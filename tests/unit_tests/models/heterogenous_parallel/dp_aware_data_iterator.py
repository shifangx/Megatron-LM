"""
DP-Aware Data Iterator for Heterogeneous Multi-Module Training

This module provides a minimal, focused implementation of DP-aware data loading
for MIMO models with heterogeneous Data Parallel configurations.

Key Principle:
    Given base_batch_size (LLM's micro-batch per DP replica), automatically:
    1. Calculate global_batch = base_batch_size × llm_dp_size
    2. Calculate encoder_batch = global_batch / encoder_dp_size
    3. Ensure DP replicas within same module get non-overlapping data
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Any, Iterator
import logging
import torch
import torch.distributed as dist
from torch.utils.data import Sampler, DataLoader
from examples.mimo.data.mock import MockVLMDataset
from megatron.core.pipeline_parallel.utils import is_pp_first_stage, is_pp_last_stage


@dataclass
class ModuleDPInfo:
    """DP configuration for a single module.
    
    Attributes:
        dp_size: Data parallel size for this module
        dp_rank: Data parallel rank within this module
        micro_batch_size: Micro-batch size for this module (samples per DP replica)
    """
    dp_size: int
    dp_rank: int
    micro_batch_size: int


def _get_dp_info_from_grid(grid) -> tuple[int, int]:
    """Get DP size and rank from grid's DP process group.
    
    Args:
        grid: HyperCommGrid with DP process group
        
    Returns:
        Tuple of (dp_size, dp_rank), or (1, 0) if no DP dimension
    """
    try:
        dp_pg = grid.get_pg("dp")
        if dp_pg is not None:
            return dp_pg.size(), dist.get_rank(dp_pg)
    except (AttributeError, KeyError):
        pass
    return 1, 0


def calculate_module_batch_sizes(
    module_to_grid_map: Dict[str, any],
    base_batch_size: int,
    llm_module_name: str = 'language_module'
) -> Dict[str, ModuleDPInfo]:
    """Calculate DP-aware batch sizes for all modules.
    
    Starting from the LLM's base_batch_size (micro-batch per DP replica),
    calculates the global batch and derives each module's micro-batch size.
    
    Args:
        module_to_grid_map: Maps module names to their HyperCommGrids
        base_batch_size: LLM's micro-batch size per DP replica
        llm_module_name: Name of the LLM module (default: 'language_module')
        
    Returns:
        Dictionary mapping module names to their ModuleDPInfo
        
    Raises:
        ValueError: If batch size calculation results in non-integer values
    """
    # Step 1: Get LLM's DP configuration
    llm_grid = module_to_grid_map[llm_module_name]
    llm_dp_size, llm_dp_rank = _get_dp_info_from_grid(llm_grid)
    
    # Step 2: Calculate global batch per microbatch
    global_batch_per_microbatch = base_batch_size * llm_dp_size
    
    # Step 3: Calculate batch sizes for all modules
    module_dp_info = {}
    
    for module_name, grid in module_to_grid_map.items():
        dp_size, dp_rank = _get_dp_info_from_grid(grid)
        
        # Calculate micro-batch size for this module
        if global_batch_per_microbatch % dp_size != 0:
            raise ValueError(
                f"Invalid DP configuration for module '{module_name}': "
                f"global_batch ({global_batch_per_microbatch}) must be divisible by "
                f"dp_size ({dp_size}). "
                f"LLM: base_batch={base_batch_size}, dp={llm_dp_size}"
            )
        
        micro_batch_size = global_batch_per_microbatch // dp_size
        
        module_dp_info[module_name] = ModuleDPInfo(
            dp_size=dp_size,
            dp_rank=dp_rank,
            micro_batch_size=micro_batch_size
        )
    
    return module_dp_info


class HeterogeneousDPBatchSampler(Sampler):
    """DP-aware batch sampler for heterogeneous multi-module training.
    
    Ensures different DP replicas within the same module receive non-overlapping
    data while respecting module-specific batch sizes.
    
    Example:
        With global_batch=8, dp_rank=0, dp_size=2, micro_batch=4:
        - DP rank 0: indices [0,2,4,6] (every 2nd sample starting from 0)
        - DP rank 1: indices [1,3,5,7] (every 2nd sample starting from 1)
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
            # Each DP replica gets every dp_size-th sample starting from dp_rank
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


class DPAwareDataIterator:
    """DP-aware data iterator with configuration logging."""
    
    def __init__(
        self,
        module_to_grid_map: Dict[str, Any],
        base_batch_size: int,
        num_microbatches: int,
        llm_module_name: str = 'language_module'
    ):
        """
        Args:
            module_to_grid_map: Maps module names to their HyperCommGrids
            base_batch_size: LLM's micro-batch size per DP replica
            num_microbatches: Number of microbatches for gradient accumulation
            llm_module_name: Name of the LLM module
        """
        self.module_to_grid_map = module_to_grid_map
        self.base_batch_size = base_batch_size
        self.num_microbatches = num_microbatches
        self.llm_module_name = llm_module_name
        
        # Calculate DP info for all modules
        self.module_dp_info = calculate_module_batch_sizes(
            module_to_grid_map, base_batch_size, llm_module_name
        )
        
        # Calculate global batch size
        llm_dp_size = self.module_dp_info[llm_module_name].dp_size
        self.global_batch_size = base_batch_size * llm_dp_size
        
        # Log configuration
        self._log_configuration()
    
    def _log_configuration(self):
        """Log the DP configuration for all modules."""
        if dist.get_rank() == 0:
            effective_global_batch = self.global_batch_size * self.num_microbatches
            
            logging.info("=" * 60)
            logging.info("DP-Aware Data Iterator Configuration")
            logging.info("=" * 60)
            logging.info(f"Base batch size (LLM per DP replica): {self.base_batch_size}")
            logging.info(f"Global batch per microbatch: {self.global_batch_size}")
            logging.info(f"Number of microbatches: {self.num_microbatches}")
            logging.info(f"Effective global batch: {effective_global_batch}")
            logging.info("-" * 60)
            
            for module_name, dp_info in self.module_dp_info.items():
                total_per_iteration = dp_info.micro_batch_size * self.num_microbatches
                logging.info(
                    f"{module_name:20s} | DP={dp_info.dp_size} | "
                    f"micro_batch={dp_info.micro_batch_size:3d} | "
                    f"total/iter={total_per_iteration:4d}"
                )
            
            logging.info("=" * 60)
    
    def get_dp_info_for_current_rank(self) -> Optional[ModuleDPInfo]:
        """Get DP info for the current rank's module.
        
        Returns:
            ModuleDPInfo for the current rank's module, or None if rank not in any grid
        """
        current_rank = dist.get_rank()
        
        # Find which module this rank belongs to
        for module_name, grid in self.module_to_grid_map.items():
            if grid.rank_offset <= current_rank < (grid.rank_offset + grid.size):
                return self.module_dp_info[module_name]
        
        return None


# ==============================================================================
# Data Iterator Functions
# ==============================================================================


def _collate_fn(
    batch: List[Dict], 
    image_seq_length: int = 1024, 
    hidden_size: int = 1024
) -> Dict[str, torch.Tensor]:
    """Collate function for the DataLoader.
    
    Args:
        batch: List of dictionaries from the dataset
        image_seq_length: Sequence length for image tokens
        hidden_size: Hidden size for the vision encoder output
        
    Returns:
        Dictionary of batched tensors
    """
    input_ids = torch.stack([item["input_ids"] for item in batch])
    labels = torch.stack([item["labels"] for item in batch])
    loss_mask = torch.stack([item["loss_mask"] for item in batch])
    position_ids = torch.stack([item["position_ids"] for item in batch])
    
    bsz = input_ids.shape[0]
    
    # Pre-allocate on pinned memory for faster GPU transfer
    # Using torch.zeros instead of randn is much faster if acceptable for testing
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
    """Recursively move tensors in nested dicts to device with non_blocking for async transfers.
    
    When pin_memory=True in DataLoader, non_blocking=True enables async GPU transfers.
    
    Args:
        data: Data to move (tensor, dict, list, or other)
        device: Target device
        
    Returns:
        Data moved to device
    """
    if isinstance(data, torch.Tensor):
        return data.to(device, non_blocking=True)
    elif isinstance(data, dict):
        return {k: move_to_device(v, device) for k, v in data.items()}
    elif isinstance(data, list):
        return [move_to_device(v, device) for v in data]
    return data


def get_data_iterator(
    module_to_grid_map: Dict[str, Any],
    base_batch_size: int,
    image_seq_length: int,
    seq_length: int,
    image_special_token_id: int,
    vocab_size: int,
    vision_hidden_size: int,
    num_microbatches: int = 1,
    encoder_module_name: str = 'images',
    llm_module_name: str = 'language_module',
    dataset_size: int = 256,
    num_workers: int = 8,
    prefetch_factor: int = 4,
):
    """Create DP-aware data iterator for heterogeneous multi-module training.
    
    This is the main entry point for creating data iterators with automatic
    DP-aware batch sizing based on module configurations.
    
    Args:
        module_to_grid_map: Maps module names to their HyperCommGrids
        base_batch_size: LLM's micro-batch size per DP replica
        image_seq_length: Sequence length for image tokens
        seq_length: Total sequence length
        image_special_token_id: Special token ID for images
        vocab_size: Vocabulary size
        vision_hidden_size: Hidden size for vision encoder
        num_microbatches: Number of microbatches (for logging only)
        encoder_module_name: Name of encoder module (default: 'images')
        llm_module_name: Name of LLM module (default: 'language_module')
        dataset_size: Size of the dataset (default: 256)
        num_workers: Number of DataLoader workers (default: 8)
        prefetch_factor: Prefetch batches per worker (default: 4)
        
    Returns:
        Data iterator configured for this rank's module, or None if rank not involved
        
    Example:
        >>> data_iterator = get_data_iterator(
        ...     module_to_grid_map={'images': encoder_grid, 'language_module': llm_grid},
        ...     base_batch_size=8,
        ...     image_seq_length=256,
        ...     seq_length=1024,
        ...     image_special_token_id=32000,
        ...     vocab_size=4000,
        ...     vision_hidden_size=768,
        ...     num_microbatches=4,
        ... )
    """
    # Create DP-aware iterator manager (logs configuration)
    dp_iterator = DPAwareDataIterator(
        module_to_grid_map, 
        base_batch_size, 
        num_microbatches, 
        llm_module_name
    )
    
    # Get DP info for current rank's module
    dp_info = dp_iterator.get_dp_info_for_current_rank()
    
    if dp_info is None:
        return None
    
    # Check if we should initialize iterator on this rank
    # Initialize on first PP stage of encoders and LLM first/last stages
    encoder_grid = module_to_grid_map.get(encoder_module_name)
    llm_grid = module_to_grid_map.get(llm_module_name)
    
    # Helper function to check if current rank is in grid
    def is_rank_in_grid(grid):
        if grid is None:
            return False
        current_rank = dist.get_rank()
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
    
    if encoder_condition or llm_condition:
        # Create dataset
        dataset = MockVLMDataset(
            size=dataset_size,
            image_size=224,
            seq_len=seq_length,
            image_seq_length=image_seq_length,
            pad_token_id=0,
            image_token_id=image_special_token_id
        )
        
        # Create DP-aware batch sampler with global batch size
        batch_sampler = HeterogeneousDPBatchSampler(
            dataset_size=len(dataset),
            global_batch_size=dp_iterator.global_batch_size,
            dp_rank=dp_info.dp_rank,
            dp_size=dp_info.dp_size,
            drop_last=True
        )
        
        # Create DataLoader with DP-aware sampler
        dataloader = DataLoader(
            dataset,
            batch_sampler=batch_sampler,
            num_workers=num_workers,
            collate_fn=lambda batch: _collate_fn(
                batch, 
                image_seq_length=image_seq_length, 
                hidden_size=vision_hidden_size
            ),
            pin_memory=True,  # Enables fast CPU->GPU transfers
            persistent_workers=True,  # Keep workers alive between epochs
            prefetch_factor=prefetch_factor,  # Prefetch batches per worker
        )
        return iter(dataloader)
    
    return None


def get_batch(data_iterator: Iterator[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
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
    else:
        input_tensor = None
    
    return input_tensor
