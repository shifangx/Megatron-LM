# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

"""Unit tests for ColocatedBridgeCommunicator."""

import logging
import os
import sys

import pytest
import torch
import torch.distributed as dist
from packaging import version

from megatron.core.hyper_comm_grid import HyperCommGrid
from megatron.core.pipeline_parallel.colocated_communicator import (
    ColocatedBridgeCommunicator,
    SliceInfo,
)

# Import utilities from bridge communicator tests
from tests.unit_tests.pipeline_parallel.test_bridge_communicator import (
    _create_transformer_block,
    _shard_and_copy_,
    _get_pg_collection_from_grid,
    _avg_params,
)
from tests.unit_tests.test_utilities import Utils

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    stream=sys.stdout,
    force=True,
)


def create_hypercomm_grid(offset=0, tp=1, cp=1, pp=1, dp=1):
    """Create a HyperCommGrid for testing."""
    if "WORLD_SIZE" not in os.environ:
        os.environ["WORLD_SIZE"] = "8"
    
    grid = HyperCommGrid(
        shape=[tp, cp, pp, dp],
        dim_names=["tp", "cp", "pp", "dp"],
        rank_offset=offset,
        backend="nccl",
    )
    # Create process groups needed by transformer blocks
    _ = grid.create_pg(["tp"])
    _ = grid.create_pg(["cp"])
    _ = grid.create_pg(["pp"])
    _ = grid.create_pg(["dp"])
    return grid


class TestColocatedCommunicatorRankMappings:
    """Test _build_rank_mappings for various configurations."""
    
    @classmethod
    def setup_class(cls):
        if not dist.is_initialized():
            dist.init_process_group(backend="nccl")
        if torch.cuda.is_available():
            torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", 0)))
    
    @pytest.mark.parametrize(
        "src_tp, src_dp, dest_tp, dest_dp, expected_src_mappings, expected_dest_mappings",
        [
            # Fan-in: DP4 → DP2 (TP2 DP4 → TP4 DP2)
            (
                2, 4,  # src: TP2 DP4
                4, 2,  # dest: TP4 DP2
                # src mappings: rank -> (dp_idx, tp_idx)
                {0: (0, 0), 1: (0, 1), 2: (1, 0), 3: (1, 1),
                 4: (2, 0), 5: (2, 1), 6: (3, 0), 7: (3, 1)},
                # dest mappings: rank -> (dp_idx, tp_idx)
                {0: (0, 0), 1: (0, 1), 2: (0, 2), 3: (0, 3),
                 4: (1, 0), 5: (1, 1), 6: (1, 2), 7: (1, 3)},
            ),
            # Fan-out: DP2 → DP4 (TP4 DP2 → TP2 DP4)
            (
                4, 2,  # src: TP4 DP2
                2, 4,  # dest: TP2 DP4
                # src mappings
                {0: (0, 0), 1: (0, 1), 2: (0, 2), 3: (0, 3),
                 4: (1, 0), 5: (1, 1), 6: (1, 2), 7: (1, 3)},
                # dest mappings
                {0: (0, 0), 1: (0, 1), 2: (1, 0), 3: (1, 1),
                 4: (2, 0), 5: (2, 1), 6: (3, 0), 7: (3, 1)},
            ),
            # Equal DP: TP4 DP2 → TP4 DP2
            (
                4, 2,
                4, 2,
                {0: (0, 0), 1: (0, 1), 2: (0, 2), 3: (0, 3),
                 4: (1, 0), 5: (1, 1), 6: (1, 2), 7: (1, 3)},
                {0: (0, 0), 1: (0, 1), 2: (0, 2), 3: (0, 3),
                 4: (1, 0), 5: (1, 1), 6: (1, 2), 7: (1, 3)},
            ),
            # Simple: TP1 DP2 → TP2 DP1 (world size 2, but we test with 8)
            (
                1, 8,  # src: TP1 DP8
                8, 1,  # dest: TP8 DP1
                {i: (i, 0) for i in range(8)},  # Each rank is its own DP replica
                {i: (0, i) for i in range(8)},  # All ranks in same DP, different TP
            ),
        ],
    )
    def test_rank_mappings(
        self, src_tp, src_dp, dest_tp, dest_dp, expected_src_mappings, expected_dest_mappings
    ):
        """Test that rank mappings are computed correctly."""
        src_grid = create_hypercomm_grid(tp=src_tp, pp=1, dp=src_dp)
        dest_grid = create_hypercomm_grid(tp=dest_tp, pp=1, dp=dest_dp)
        
        comm = ColocatedBridgeCommunicator(
            src_grid=src_grid,
            dest_grid=dest_grid,
            src_module_name="encoder",
            dest_module_name="llm",
        )
        
        # Verify src mappings
        assert comm.rank_to_src_pos == expected_src_mappings, (
            f"Src mapping mismatch: expected {expected_src_mappings}, got {comm.rank_to_src_pos}"
        )
        
        # Verify dest mappings
        assert comm.rank_to_dest_pos == expected_dest_mappings, (
            f"Dest mapping mismatch: expected {expected_dest_mappings}, got {comm.rank_to_dest_pos}"
        )


class TestColocatedCommunicatorAllGatherGroups:
    """Test _build_all_gather_groups for fan-in cases."""
    
    @classmethod
    def setup_class(cls):
        if not dist.is_initialized():
            dist.init_process_group(backend="nccl")
        if torch.cuda.is_available():
            torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", 0)))
    
    @pytest.mark.parametrize(
        "src_tp, src_dp, dest_tp, dest_dp, expected_groups",
        [
            # Fan-in: TP2 DP4 → TP4 DP2
            # scale = 4/2 = 2, so 2 src DP replicas per dest DP replica
            # Groups are formed by same src_tp position, same dest_dp replica
            (
                2, 4,  # src: TP2 DP4
                4, 2,  # dest: TP4 DP2
                [
                    # dest_dp=0: src_dp 0,1
                    [0, 2],  # src_tp=0: rank 0 (dp0,tp0) and rank 2 (dp1,tp0)
                    [1, 3],  # src_tp=1: rank 1 (dp0,tp1) and rank 3 (dp1,tp1)
                    # dest_dp=1: src_dp 2,3
                    [4, 6],  # src_tp=0: rank 4 (dp2,tp0) and rank 6 (dp3,tp0)
                    [5, 7],  # src_tp=1: rank 5 (dp2,tp1) and rank 7 (dp3,tp1)
                ],
            ),
            # Fan-in: TP1 DP8 → TP8 DP1
            # scale = 8/1 = 8, all src DP replicas contribute to single dest DP
            (
                1, 8,  # src: TP1 DP8
                8, 1,  # dest: TP8 DP1
                [
                    # dest_dp=0: all src_dp 0-7, src_tp=0
                    [0, 1, 2, 3, 4, 5, 6, 7],
                ],
            ),
            # Fan-in: TP2 DP4 → TP8 DP1
            # scale = 4/1 = 4, all 4 src DP replicas contribute to single dest DP
            (
                2, 4,  # src: TP2 DP4
                8, 1,  # dest: TP8 DP1
                [
                    # dest_dp=0: all src_dp 0-3
                    [0, 2, 4, 6],  # src_tp=0
                    [1, 3, 5, 7],  # src_tp=1
                ],
            ),
            # Fan-in: TP4 DP2 → TP8 DP1
            # scale = 2/1 = 2
            (
                4, 2,  # src: TP4 DP2  
                8, 1,  # dest: TP8 DP1
                [
                    # dest_dp=0: src_dp 0,1
                    [0, 4],  # src_tp=0
                    [1, 5],  # src_tp=1
                    [2, 6],  # src_tp=2
                    [3, 7],  # src_tp=3
                ],
            ),
        ],
    )
    def test_all_gather_groups(self, src_tp, src_dp, dest_tp, dest_dp, expected_groups):
        """Test that all-gather groups are formed correctly for fan-in."""
        src_grid = create_hypercomm_grid(tp=src_tp, pp=1, dp=src_dp)
        dest_grid = create_hypercomm_grid(tp=dest_tp, pp=1, dp=dest_dp)
        
        comm = ColocatedBridgeCommunicator(
            src_grid=src_grid,
            dest_grid=dest_grid,
            src_module_name="encoder",
            dest_module_name="llm",
        )
        
        # Should be fan-in
        assert comm.is_fan_in(), f"Expected fan-in for DP{src_dp}→DP{dest_dp}"
        
        # Verify groups
        assert comm.all_gather_group_ranks == expected_groups, (
            f"Group mismatch: expected {expected_groups}, got {comm.all_gather_group_ranks}"
        )
    
    def test_fan_out_no_groups(self):
        """Test that fan-out case doesn't create all-gather groups."""
        # Fan-out: TP4 DP2 → TP2 DP4
        src_grid = create_hypercomm_grid(tp=4, pp=1, dp=2)
        dest_grid = create_hypercomm_grid(tp=2, pp=1, dp=4)
        
        comm = ColocatedBridgeCommunicator(
            src_grid=src_grid,
            dest_grid=dest_grid,
            src_module_name="encoder",
            dest_module_name="llm",
        )
        
        assert comm.is_fan_out()
        assert comm.all_gather_group_ranks == []
        assert comm.all_gather_pg is None


class TestColocatedCommunicatorSliceInfo:
    """Test get_slice_info for fan-out and fan-in cases."""
    
    @classmethod
    def setup_class(cls):
        if not dist.is_initialized():
            dist.init_process_group(backend="nccl")
        if torch.cuda.is_available():
            torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", 0)))
    
    @pytest.mark.parametrize(
        "src_tp, src_dp, dest_tp, dest_dp, batch_size, rank, expected_slice",
        [
            # Fan-out: TP4 DP2 → TP2 DP4, batch=16
            # src DP0 (ranks 0-3) splits into dest DP0,DP1
            # src DP1 (ranks 4-7) splits into dest DP2,DP3
            (4, 2, 2, 4, 16, 0, SliceInfo(start=0, size=8)),   # dest_dp=0, slot=0
            (4, 2, 2, 4, 16, 1, SliceInfo(start=0, size=8)),   # dest_dp=0, slot=0
            (4, 2, 2, 4, 16, 2, SliceInfo(start=8, size=8)),   # dest_dp=1, slot=1
            (4, 2, 2, 4, 16, 3, SliceInfo(start=8, size=8)),   # dest_dp=1, slot=1
            (4, 2, 2, 4, 16, 4, SliceInfo(start=0, size=8)),   # dest_dp=2, slot=0
            (4, 2, 2, 4, 16, 5, SliceInfo(start=0, size=8)),   # dest_dp=2, slot=0
            (4, 2, 2, 4, 16, 6, SliceInfo(start=8, size=8)),   # dest_dp=3, slot=1
            (4, 2, 2, 4, 16, 7, SliceInfo(start=8, size=8)),   # dest_dp=3, slot=1
            
            # Fan-out: TP1 DP2 → TP1 DP8 (hypothetical with world=8)
            # Actually this doesn't work with world=8, skip
            
            # Fan-out: TP2 DP4 → TP1 DP8 won't work either, skip
        ],
    )
    def test_fan_out_slice_info(
        self, src_tp, src_dp, dest_tp, dest_dp, batch_size, rank, expected_slice
    ):
        """Test slice info for fan-out case."""
        if rank != dist.get_rank():
            pytest.skip(f"Test only for rank {rank}")
        
        src_grid = create_hypercomm_grid(tp=src_tp, pp=1, dp=src_dp)
        dest_grid = create_hypercomm_grid(tp=dest_tp, pp=1, dp=dest_dp)
        
        comm = ColocatedBridgeCommunicator(
            src_grid=src_grid,
            dest_grid=dest_grid,
            src_module_name="encoder",
            dest_module_name="llm",
        )
        
        assert comm.is_fan_out()
        
        slice_info = comm.get_slice_info(batch_size)
        assert slice_info.start == expected_slice.start, (
            f"Rank {rank}: start mismatch, expected {expected_slice.start}, got {slice_info.start}"
        )
        assert slice_info.size == expected_slice.size, (
            f"Rank {rank}: size mismatch, expected {expected_slice.size}, got {slice_info.size}"
        )
    
    @pytest.mark.parametrize(
        "src_tp, src_dp, dest_tp, dest_dp, batch_size, rank, expected_slice",
        [
            # Fan-in: TP2 DP4 → TP4 DP2, batch=16 (after gather)
            # Backward slice: each rank keeps its src_dp portion
            # dest DP0 gathers from src DP0,1 → ranks keep slot 0 or 1
            (2, 4, 4, 2, 16, 0, SliceInfo(start=0, size=8)),   # src_dp=0, slot=0
            (2, 4, 4, 2, 16, 1, SliceInfo(start=0, size=8)),   # src_dp=0, slot=0
            (2, 4, 4, 2, 16, 2, SliceInfo(start=8, size=8)),   # src_dp=1, slot=1
            (2, 4, 4, 2, 16, 3, SliceInfo(start=8, size=8)),   # src_dp=1, slot=1
            (2, 4, 4, 2, 16, 4, SliceInfo(start=0, size=8)),   # src_dp=2, slot=0
            (2, 4, 4, 2, 16, 5, SliceInfo(start=0, size=8)),   # src_dp=2, slot=0
            (2, 4, 4, 2, 16, 6, SliceInfo(start=8, size=8)),   # src_dp=3, slot=1
            (2, 4, 4, 2, 16, 7, SliceInfo(start=8, size=8)),   # src_dp=3, slot=1
        ],
    )
    def test_fan_in_slice_info(
        self, src_tp, src_dp, dest_tp, dest_dp, batch_size, rank, expected_slice
    ):
        """Test slice info for fan-in backward (which portion of gradient to keep)."""
        if rank != dist.get_rank():
            pytest.skip(f"Test only for rank {rank}")
        
        src_grid = create_hypercomm_grid(tp=src_tp, pp=1, dp=src_dp)
        dest_grid = create_hypercomm_grid(tp=dest_tp, pp=1, dp=dest_dp)
        
        comm = ColocatedBridgeCommunicator(
            src_grid=src_grid,
            dest_grid=dest_grid,
            src_module_name="encoder",
            dest_module_name="llm",
        )
        
        assert comm.is_fan_in()
        
        slice_info = comm.get_slice_info(batch_size)
        assert slice_info.start == expected_slice.start, (
            f"Rank {rank}: start mismatch, expected {expected_slice.start}, got {slice_info.start}"
        )
        assert slice_info.size == expected_slice.size, (
            f"Rank {rank}: size mismatch, expected {expected_slice.size}, got {slice_info.size}"
        )
    
    def test_equal_dp_slice_info(self):
        """Test slice info for equal DP case (no slicing needed)."""
        src_grid = create_hypercomm_grid(tp=4, pp=1, dp=2)
        dest_grid = create_hypercomm_grid(tp=4, pp=1, dp=2)
        
        comm = ColocatedBridgeCommunicator(
            src_grid=src_grid,
            dest_grid=dest_grid,
            src_module_name="encoder",
            dest_module_name="llm",
        )
        
        assert comm.is_equal_dp()
        
        batch_size = 16
        slice_info = comm.get_slice_info(batch_size)
        
        # Equal DP: keep entire batch
        assert slice_info.start == 0
        assert slice_info.size == batch_size

class TestColocatedCommunicatorGolden:
    """Golden tests for forward and backward pass correctness.
    
    These tests verify that colocated communication produces numerically
    correct results by comparing against a reference computation.
    """
    
    @classmethod
    def setup_class(cls):
        if not dist.is_initialized():
            dist.init_process_group(backend="nccl")
        if torch.cuda.is_available():
            torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", 0)))
        
        world_size = dist.get_world_size()
        if world_size != 8:
            pytest.skip(
                f"These tests require 8 GPUs, but only {world_size} are available.",
                allow_module_level=True,
            )
    
    @classmethod
    def teardown_class(cls):
        Utils.destroy_model_parallel()
    
    @pytest.mark.skipif(
        version.parse(torch.__version__) < version.parse('2.3.0'),
        reason="Feature requires PyTorch 2.3 or later",
    )
    @pytest.mark.parametrize(
        "src_tp, src_dp, dest_tp, dest_dp",
        [
            (4, 2, 2, 4),  # Fan-out: DP2 → DP4
            (2, 4, 4, 2),  # Fan-in: DP4 → DP2
            (4, 2, 4, 2),  # Equal DP
        ],
    )
    def test_golden_forward_backward(self, src_tp, src_dp, dest_tp, dest_dp):
        """Golden test: verify forward and backward match reference computation.
        
        Strategy:
        1. Create global input tensor (same on all ranks)
        2. Reference: Run full batch through TP1 blocks
        3. Colocated: Each rank processes its DP slice, communicates, processes
        4. Compare: Each rank's output should match corresponding slice of reference
        5. Backward: Verify gradients are correct
        """
        # Disable non-deterministic ops
        os.environ["NVTE_ALLOW_NONDETERMINISTIC_ALGO"] = "0"
        os.environ["NVTE_FLASH_ATTN"] = "0"
        os.environ["NVTE_FUSED_ATTN"] = "0"
        
        hidden_size = 1024
        sequence_length = 8
        global_batch_size = 8  # Total batch across all DP replicas
        dtype = torch.float32
        current_rank = dist.get_rank()
        
        # Deterministic setup
        torch.manual_seed(12345)
        torch.cuda.manual_seed(12345)
        
        Utils.initialize_model_parallel(
            tensor_model_parallel_size=1, create_gloo_process_groups=False
        )
        
        # Create global input - SAME on all ranks for reference comparison
        global_input = torch.randn(
            (sequence_length, global_batch_size, hidden_size),
            device="cuda",
            dtype=dtype,
        )
        global_input.requires_grad_(True)
        
        # ========== REFERENCE COMPUTATION ==========
        # Create TP1 reference blocks (all ranks compute the same thing)
        ref_grid = create_hypercomm_grid(tp=1, pp=1, dp=8)
        ref_pg = _get_pg_collection_from_grid(ref_grid)
        
        torch.manual_seed(42)
        ref_block1 = _create_transformer_block(
            dtype=dtype, hidden_size=hidden_size, pg_collection=ref_pg
        )
        _avg_params(ref_block1, ref_grid.get_pg("dp"))
        
        torch.manual_seed(42)
        ref_block2 = _create_transformer_block(
            dtype=dtype, hidden_size=hidden_size, pg_collection=ref_pg
        )
        _avg_params(ref_block2, ref_grid.get_pg("dp"))
        
        # Reference forward
        ref_out1 = ref_block1(hidden_states=global_input, attention_mask=None)
        ref_out2 = ref_block2(hidden_states=ref_out1, attention_mask=None)
        
        # Reference backward
        ref_loss = ref_out2.sum()
        ref_loss.backward()
        ref_grad = global_input.grad.clone()
        global_input.grad = None
        
        # ========== COLOCATED COMPUTATION ==========
        # Create colocated grids
        src_grid = create_hypercomm_grid(tp=src_tp, pp=1, dp=src_dp)
        dest_grid = create_hypercomm_grid(tp=dest_tp, pp=1, dp=dest_dp)
        
        # Create blocks with same weights as reference
        src_pg = _get_pg_collection_from_grid(src_grid)
        torch.manual_seed(42)
        src_block = _create_transformer_block(
            dtype=dtype, hidden_size=hidden_size, pg_collection=src_pg
        )
        _shard_and_copy_(ref_block1, src_block, src_tp, src_pg.tp.rank())
        
        dest_pg = _get_pg_collection_from_grid(dest_grid)
        torch.manual_seed(42)
        dest_block = _create_transformer_block(
            dtype=dtype, hidden_size=hidden_size, pg_collection=dest_pg
        )
        _shard_and_copy_(ref_block2, dest_block, dest_tp, dest_pg.tp.rank())
        
        # Create communicator
        comm = ColocatedBridgeCommunicator(
            src_grid=src_grid,
            dest_grid=dest_grid,
            src_module_name="encoder",
            dest_module_name="llm",
            dim_mapping={'s': 0, 'b': 1, 'h': 2},
        )
        
        dist.barrier()
        
        # Get this rank's input slice based on src DP position
        src_dp_idx, _ = comm.rank_to_src_pos[current_rank]
        per_src_dp_batch = global_batch_size // src_dp
        my_input = global_input[:, src_dp_idx * per_src_dp_batch:(src_dp_idx + 1) * per_src_dp_batch, :].clone().detach()
        my_input.requires_grad_(True)
        
        # Colocated forward
        src_out = src_block(hidden_states=my_input, attention_mask=None)
        communicated = comm.communicate(src_out)
        dest_out = dest_block(hidden_states=communicated, attention_mask=None)
        
        # ========== FORWARD VERIFICATION ==========
        # Get expected output slice for this rank based on dest DP position
        dest_dp_idx, _ = comm.rank_to_dest_pos[current_rank]
        per_dest_dp_batch = global_batch_size // dest_dp
        expected_out = ref_out2[:, dest_dp_idx * per_dest_dp_batch:(dest_dp_idx + 1) * per_dest_dp_batch, :]
        
        # Forward should match (1e-3 tolerance for TP all-reduce numerical errors in transformer blocks)
        torch.testing.assert_close(
            dest_out, expected_out, rtol=1e-3, atol=1e-3,
            msg=f"Rank {current_rank}: Forward output mismatch"
        )
        
        logging.info(
            f"Rank {current_rank}: Forward PASSED - "
            f"output shape {dest_out.shape}, expected slice [{dest_dp_idx * per_dest_dp_batch}:{(dest_dp_idx + 1) * per_dest_dp_batch}]"
        )
        
        # ========== BACKWARD VERIFICATION ==========
        # Colocated backward
        colocated_loss = dest_out.sum()
        colocated_loss.backward()
        
        # Get expected gradient slice for this rank's input
        expected_grad = ref_grad[:, src_dp_idx * per_src_dp_batch:(src_dp_idx + 1) * per_src_dp_batch, :]
        
        # For fan-out backward: gradients are partial (zeros outside slice)
        # The test verifies the structure is correct
        # For fan-in backward: gradients are sliced from gathered gradient
        
        assert my_input.grad is not None, f"Rank {current_rank}: No gradient computed"
        
        if comm.is_fan_out():
            # Fan-out: compare each rank's gradient slice directly to reference
            # (avoids all-reduce which could mask slice position bugs)
            slice_info = comm.get_slice_info(per_src_dp_batch)
            my_grad_slice = my_input.grad[:, slice_info.start:slice_info.start + slice_info.size, :]
            
            # This rank processed dest_dp_idx portion of global batch
            dest_dp_idx, _ = comm.rank_to_dest_pos[current_rank]
            per_dest_dp_batch = global_batch_size // dest_dp
            expected_slice = ref_grad[:, dest_dp_idx * per_dest_dp_batch:(dest_dp_idx + 1) * per_dest_dp_batch, :]
            
            torch.testing.assert_close(
                my_grad_slice, expected_slice, rtol=1e-5, atol=1e-5,
                msg=f"Rank {current_rank}: Backward gradient slice mismatch (fan-out)"
            )
        else:
            # Equal DP and Fan-in: gradient matches reference directly
            torch.testing.assert_close(
                my_input.grad, expected_grad, rtol=1e-5, atol=1e-5,
                msg=f"Rank {current_rank}: Backward gradient mismatch"
            )
        
        logging.info(f"Rank {current_rank}: Backward PASSED - gradient matches reference")
        
        Utils.destroy_model_parallel()