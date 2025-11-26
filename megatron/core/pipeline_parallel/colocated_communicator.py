# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

"""Communicator for colocated modules with different TP/DP configurations.

This module provides communication primitives for MIMO models where encoder and LLM
are colocated on the same GPUs but have different tensor/data parallelism settings.
"""

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    pass  # For forward references if needed

import torch
import torch.distributed as dist

from megatron.core.hyper_comm_grid import HyperCommGrid


@dataclass
class SliceInfo:
    """Information for slicing tensors along batch dimension."""
    start: int
    size: int


class ColocatedBridgeCommunicator:
    """Communicator for colocated modules with different TP/DP configurations.
    
    Handles tensor transfer between modules that:
    - Run on the same GPU ranks (colocated)
    - Have PP=1 (no pipeline parallelism)
    - May have different TP/DP configurations
    
    Key optimizations:
    - Fan-out (src_dp < dest_dp): Zero communication - local slice
    - Fan-in (src_dp > dest_dp): Parallel all-gathers instead of P2P + broadcast
    
    Args:
        src_grid: HyperCommGrid for source module
        dest_grid: HyperCommGrid for destination module
        src_module_name: Name of source module (for logging)
        dest_module_name: Name of destination module (for logging)
        dim_mapping: Mapping of logical dims to tensor axes.
                    Defaults to {'b': 0, 's': 1, 'h': 2}
    """
    
    def __init__(
        self,
        src_grid: HyperCommGrid,
        dest_grid: HyperCommGrid,
        src_module_name: str = "src",
        dest_module_name: str = "dest",
        dim_mapping: Optional[Dict[str, int]] = None,
    ):
        self.src_grid = src_grid
        self.dest_grid = dest_grid
        self.src_module_name = src_module_name
        self.dest_module_name = dest_module_name
        self.dim_mapping = dim_mapping or {'b': 0, 's': 1, 'h': 2}
        self.current_rank = dist.get_rank()
        
        # Validate grids meet colocated requirements
        self._validate_grids()
        
        # Extract parallelism info
        self._extract_parallelism_info()
        
        # Build rank to position mappings
        self._build_rank_mappings()
        
        # Build communication groups based on case
        self.all_gather_pg = None
        self.all_gather_group_ranks: List[List[int]] = []
        
        if self.dp_scale_factor > 1:  # Fan-in case
            self._build_all_gather_groups()
        
        logging.info(
            f"[Rank {self.current_rank}] ColocatedBridgeCommunicator initialized: "
            f"{src_module_name}({self.src_tp_size}TP/{self.src_dp_size}DP) → "
            f"{dest_module_name}({self.dest_tp_size}TP/{self.dest_dp_size}DP), "
            f"scale_factor={self.dp_scale_factor}"
        )
    
    def _validate_grids(self):
        """Validate that grids are properly configured for colocated communication."""
        # Both grids must span same ranks
        if self.src_grid.size != self.dest_grid.size:
            raise ValueError(
                f"Grids must span same number of ranks: "
                f"src={self.src_grid.size}, dest={self.dest_grid.size}"
            )
        
        if self.src_grid.rank_offset != self.dest_grid.rank_offset:
            raise ValueError(
                f"Grids must have same rank offset: "
                f"src={self.src_grid.rank_offset}, dest={self.dest_grid.rank_offset}"
            )
        
        # PP must be 1 for both (colocated assumption)
        # TODO: ykarnati - what if grid dont have pp ? 
        # we check for pp only if it exists in the grid
        src_pp_idx = self.src_grid.dim_names.index('pp')
        dest_pp_idx = self.dest_grid.dim_names.index('pp')
        src_pp = self.src_grid.shape[src_pp_idx]
        dest_pp = self.dest_grid.shape[dest_pp_idx]
        
        if src_pp != 1:
            raise ValueError(f"Source PP must be 1 for colocated, got {src_pp}")
        if dest_pp != 1:
            raise ValueError(f"Dest PP must be 1 for colocated, got {dest_pp}")
        
        # DP sizes must be evenly divisible
        src_dp_idx = self.src_grid.dim_names.index('dp')
        dest_dp_idx = self.dest_grid.dim_names.index('dp')
        src_dp = self.src_grid.shape[src_dp_idx]
        dest_dp = self.dest_grid.shape[dest_dp_idx]
        
        if src_dp % dest_dp != 0 and dest_dp % src_dp != 0:
            raise ValueError(
                f"DP sizes must be evenly divisible: src_dp={src_dp}, dest_dp={dest_dp}"
            )
    
    def _extract_parallelism_info(self):
        """Extract TP/DP sizes from grids."""
        self.src_tp_size = self.src_grid.shape[self.src_grid.dim_names.index('tp')]
        self.src_dp_size = self.src_grid.shape[self.src_grid.dim_names.index('dp')]
        self.dest_tp_size = self.dest_grid.shape[self.dest_grid.dim_names.index('tp')]
        self.dest_dp_size = self.dest_grid.shape[self.dest_grid.dim_names.index('dp')]
        
        # Scale factor determines fan-in vs fan-out
        # > 1: fan-in (src has more DP replicas, need to gather)
        # < 1: fan-out (src has fewer DP replicas, need to split/slice)
        # = 1: equal DP
        self.dp_scale_factor = self.src_dp_size / self.dest_dp_size
    
    def _build_rank_mappings(self):
        """Build mappings from rank to DP/TP positions in both grids."""
        self.rank_to_src_pos: Dict[int, Tuple[int, int]] = {}  # rank -> (dp_idx, tp_idx)
        self.rank_to_dest_pos: Dict[int, Tuple[int, int]] = {}
        
        # Map ranks to src grid positions
        # _gen_rank_enum(['tp']) gives ranks grouped by same DP, varying TP
        src_tp_groups = self.src_grid._gen_rank_enum(['tp'])
        for dp_idx, tp_group in enumerate(src_tp_groups):
            for tp_idx, rank in enumerate(tp_group):
                self.rank_to_src_pos[rank] = (dp_idx, tp_idx)
        
        # Map ranks to dest grid positions
        dest_tp_groups = self.dest_grid._gen_rank_enum(['tp'])
        for dp_idx, tp_group in enumerate(dest_tp_groups):
            for tp_idx, rank in enumerate(tp_group):
                self.rank_to_dest_pos[rank] = (dp_idx, tp_idx)
        
        logging.debug(
            f"[Rank {self.current_rank}] Position mappings: "
            f"src={self.rank_to_src_pos.get(self.current_rank)}, "
            f"dest={self.rank_to_dest_pos.get(self.current_rank)}"
        )
    
    def _build_all_gather_groups(self):
        """Build all-gather process groups for fan-in case.
        
        Groups contain ranks with:
        - Same src TP position (so they have different data from different src DP replicas)
        - Same dest DP replica (so they need combined data)
        """
        scale = int(self.dp_scale_factor)
        all_groups: List[List[int]] = []
        
        # For each dest DP replica
        for dest_dp_idx in range(self.dest_dp_size):
            # Which src DP replicas contribute to this dest DP replica?
            src_dp_start = dest_dp_idx * scale
            src_dp_end = src_dp_start + scale
            src_dp_indices = range(src_dp_start, src_dp_end)
            
            # For each src TP position, create a group
            for src_tp_idx in range(self.src_tp_size):
                group_ranks = []
                
                # Find ranks with this (src_dp, src_tp) position
                for src_dp_idx in src_dp_indices:
                    for rank, (dp, tp) in self.rank_to_src_pos.items():
                        if dp == src_dp_idx and tp == src_tp_idx:
                            group_ranks.append(rank)
                            break
                
                all_groups.append(sorted(group_ranks))
        
        self.all_gather_group_ranks = all_groups
        
        # Create process groups
        self.all_gather_pg, _ = dist.new_subgroups_by_enumeration(
            all_groups, backend='nccl'
        )
        
        # Find which group current rank belongs to
        self._my_all_gather_group_idx = None
        for idx, group_ranks in enumerate(all_groups):
            if self.current_rank in group_ranks:
                self._my_all_gather_group_idx = idx
                break
        
        logging.debug(
            f"[Rank {self.current_rank}] All-gather groups: {all_groups}, "
            f"my_group_idx={self._my_all_gather_group_idx}"
        )
    
    def get_all_gather_group(self) -> Optional[dist.ProcessGroup]:
        """Get the all-gather process group for current rank (fan-in case)."""
        return self.all_gather_pg
    
    def get_all_gather_world_size(self) -> int:
        """Get the world size of the all-gather group."""
        if self.all_gather_pg is None:
            return 1
        return dist.get_world_size(self.all_gather_pg)
    
    def get_slice_info(self, batch_size: int) -> SliceInfo:
        """Get slice info for current rank based on communication pattern.
        
        For fan-out: Which slice of src tensor this rank should keep for dest.
        For fan-in backward: Which slice of gradient this rank should keep.
        
        Args:
            batch_size: Total batch size of the tensor being sliced
            
        Returns:
            SliceInfo with start index and size for slicing along batch dim
        """
        if self.dp_scale_factor < 1:
            # Fan-out: slice based on dest DP position within src DP replica
            return self._get_fan_out_slice_info(batch_size)
        elif self.dp_scale_factor > 1:
            # Fan-in: slice based on src DP position within dest DP replica
            return self._get_fan_in_slice_info(batch_size)
        else:
            # Equal DP: no slicing needed
            return SliceInfo(start=0, size=batch_size)
    
    def _get_fan_out_slice_info(self, batch_size: int) -> SliceInfo:
        """Get slice info for fan-out case.
        
        Each rank determines which portion of the src tensor to keep
        based on its dest DP position.
        """
        src_dp_idx, _ = self.rank_to_src_pos[self.current_rank]
        dest_dp_idx, _ = self.rank_to_dest_pos[self.current_rank]
        
        # How many dest DP replicas per src DP replica?
        scale = int(1 / self.dp_scale_factor)  # e.g., DP2→DP4 gives scale=2
        
        # Which "slot" within the src DP is this dest DP?
        slot = dest_dp_idx % scale
        
        # Calculate slice
        slice_size = batch_size // scale
        start = slot * slice_size
        
        return SliceInfo(start=start, size=slice_size)
    
    def _get_fan_in_slice_info(self, batch_size: int) -> SliceInfo:
        """Get slice info for fan-in backward (which portion of gradient to keep).
        
        Each rank keeps the portion corresponding to its src DP position.
        """
        src_dp_idx, _ = self.rank_to_src_pos[self.current_rank]
        dest_dp_idx, _ = self.rank_to_dest_pos[self.current_rank]
        
        scale = int(self.dp_scale_factor)  # e.g., DP4→DP2 gives scale=2
        
        # Which slot within the dest DP is this src DP?
        slot = src_dp_idx % scale
        
        # Calculate slice
        slice_size = batch_size // scale
        start = slot * slice_size
        
        return SliceInfo(start=start, size=slice_size)
    
    def is_fan_out(self) -> bool:
        """Return True if this is a fan-out case (src_dp < dest_dp)."""
        return self.dp_scale_factor < 1
    
    def is_fan_in(self) -> bool:
        """Return True if this is a fan-in case (src_dp > dest_dp)."""
        return self.dp_scale_factor > 1
    
    def is_equal_dp(self) -> bool:
        """Return True if src and dest have equal DP."""
        return self.dp_scale_factor == 1
    
    def communicate(self, tensor: torch.Tensor) -> torch.Tensor:
        """Transfer tensor from src module parallelism to dest module parallelism.
        
        This is the main entry point. Uses autograd-aware operations.
        
        Args:
            tensor: Input tensor from src module
            
        Returns:
            Output tensor for dest module with correct batch partitioning
        """
        return _ColocatedCommunicate.apply(tensor, self)


class _ColocatedCommunicate(torch.autograd.Function):
    """Autograd function for colocated communication.
    
    Handles both fan-out and fan-in cases with proper backward pass.
    
    Fan-out (src_dp < dest_dp):
        Forward: Local slice (zero communication)
        Backward: Zero-padded gradient (DP all-reduce combines them)
    
    Fan-in (src_dp > dest_dp):
        Forward: Parallel all-gather
        Backward: Local slice (zero communication)
    """
    
    @staticmethod
    def forward(
        ctx,
        tensor: torch.Tensor,
        comm: 'ColocatedBridgeCommunicator',
    ) -> torch.Tensor:
        """Forward pass: transfer tensor from src to dest parallelism."""
        ctx.comm = comm
        ctx.batch_dim = comm.dim_mapping['b']
        batch_size = tensor.shape[ctx.batch_dim]
        
        if comm.is_fan_out():
            # Fan-out: local slice - zero communication
            ctx.input_batch_size = batch_size
            slice_info = comm.get_slice_info(batch_size)
            return tensor.narrow(ctx.batch_dim, slice_info.start, slice_info.size).contiguous()
        
        elif comm.is_fan_in():
            # Fan-in: parallel all-gather
            ctx.output_batch_size = batch_size * comm.get_all_gather_world_size()
            
            group = comm.get_all_gather_group()
            world_size = comm.get_all_gather_world_size()
            
            # All-gather along batch dimension
            gathered_list = [torch.empty_like(tensor) for _ in range(world_size)]
            dist.all_gather(gathered_list, tensor.contiguous(), group=group)
            
            return torch.cat(gathered_list, dim=ctx.batch_dim)
        
        else:
            # Equal DP: no-op
            return tensor.contiguous()
    
    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> Tuple[torch.Tensor, None]:
        """Backward pass: inverse of forward."""
        comm = ctx.comm
        batch_dim = ctx.batch_dim
        
        if comm.is_fan_out():
            # Fan-out backward: create zero-padded gradient
            # The encoder's DP all-reduce will sum partial gradients
            grad_input = torch.zeros(
                *grad_output.shape[:batch_dim],
                ctx.input_batch_size,
                *grad_output.shape[batch_dim + 1:],
                dtype=grad_output.dtype,
                device=grad_output.device,
            )
            slice_info = comm.get_slice_info(ctx.input_batch_size)
            grad_input.narrow(batch_dim, slice_info.start, slice_info.size).copy_(grad_output)
            return grad_input, None
        
        elif comm.is_fan_in():
            # Fan-in backward: local slice - zero communication
            output_batch_size = grad_output.shape[batch_dim]
            slice_info = comm.get_slice_info(output_batch_size)
            grad_input = grad_output.narrow(
                batch_dim, slice_info.start, slice_info.size
            ).contiguous()
            return grad_input, None
        
        else:
            # Equal DP: no-op
            return grad_output.contiguous(), None
