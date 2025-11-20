"""
Test suite for DP-aware data iterator

Run with:
python -m torch.distributed.run --nproc_per_node=8 tests/unit_tests/models/heterogenous_parallel/test_dp_iterator.py
"""

import torch
import torch.distributed as dist
from tests.unit_tests.test_utilities import Utils
from tests.unit_tests.pipeline_parallel.test_multimodule_schedules import create_hypercomm_grid
from tests.unit_tests.models.heterogenous_parallel.dp_aware_data_iterator import (
    extract_dp_from_grid,
    calculate_module_batch_sizes,
    get_module_dp_info_for_rank,
    HeterogeneousDPBatchSampler,
    log_dp_configuration,
)


def test_scenario_1():
    """Test Scenario 1: Encoder DP=2, LLM DP=1"""
    print("\n" + "="*60)
    print("Test Scenario 1: Encoder DP=2, LLM DP=1")
    print("="*60)
    
    # Create grids
    encoder_grid = create_hypercomm_grid(offset=0, tp=2, cp=1, pp=1, dp=2)
    llm_grid = create_hypercomm_grid(offset=4, tp=2, cp=1, pp=2, dp=1)
    
    module_to_grid_map = {
        'images': encoder_grid,
        'language_module': llm_grid,
    }
    
    base_batch_size = 8
    num_microbatches = 4
    
    # Log configuration
    log_dp_configuration(module_to_grid_map, base_batch_size, num_microbatches)
    
    # Calculate batch sizes
    dp_info_map = calculate_module_batch_sizes(module_to_grid_map, base_batch_size)
    
    # Verify calculations
    rank = dist.get_rank()
    
    if rank == 0:
        # Encoder should have micro_batch=4 (global_batch=8 / encoder_dp=2)
        assert dp_info_map['images'].micro_batch_size == 4, \
            f"Expected encoder micro_batch=4, got {dp_info_map['images'].micro_batch_size}"
        
        # LLM should have micro_batch=8 (global_batch=8 / llm_dp=1)
        assert dp_info_map['language_module'].micro_batch_size == 8, \
            f"Expected llm micro_batch=8, got {dp_info_map['language_module'].micro_batch_size}"
        
        print("✓ Scenario 1: Batch size calculations correct")
    
    dist.barrier()


def test_scenario_2():
    """Test Scenario 2: Encoder DP=1, LLM DP=4"""
    print("\n" + "="*60)
    print("Test Scenario 2: Encoder DP=1, LLM DP=4")
    print("="*60)
    
    # Create grids
    encoder_grid = create_hypercomm_grid(offset=0, tp=1, cp=1, pp=1, dp=1)
    llm_grid = create_hypercomm_grid(offset=1, tp=1, cp=1, pp=1, dp=4)
    
    # Only test with first 5 ranks
    if dist.get_rank() >= 5:
        dist.barrier()
        return
    
    module_to_grid_map = {
        'images': encoder_grid,
        'language_module': llm_grid,
    }
    
    base_batch_size = 16
    num_microbatches = 2
    
    # Log configuration
    log_dp_configuration(module_to_grid_map, base_batch_size, num_microbatches)
    
    # Calculate batch sizes
    dp_info_map = calculate_module_batch_sizes(module_to_grid_map, base_batch_size)
    
    # Verify calculations
    rank = dist.get_rank()
    
    if rank == 0:
        # Encoder should have micro_batch=64 (global_batch=64 / encoder_dp=1)
        assert dp_info_map['images'].micro_batch_size == 64, \
            f"Expected encoder micro_batch=64, got {dp_info_map['images'].micro_batch_size}"
        
        # LLM should have micro_batch=16 (global_batch=64 / llm_dp=4)
        assert dp_info_map['language_module'].micro_batch_size == 16, \
            f"Expected llm micro_batch=16, got {dp_info_map['language_module'].micro_batch_size}"
        
        print("✓ Scenario 2: Batch size calculations correct")
    
    dist.barrier()


def test_dp_sampler():
    """Test DP-aware batch sampler"""
    print("\n" + "="*60)
    print("Test DP-Aware Batch Sampler")
    print("="*60)
    
    # Simulate encoder with DP=2, global batch size of 8
    dataset_size = 32
    global_batch_size = 8
    dp_size = 2
    dp_rank = 0
    
    sampler = HeterogeneousDPBatchSampler(
        dataset_size=dataset_size,
        global_batch_size=global_batch_size,
        dp_rank=dp_rank,
        dp_size=dp_size,
        drop_last=True
    )
    
    rank = dist.get_rank()
    
    if rank == 0:
        batches = list(sampler)
        print(f"\nEncoder DP Rank 0 batches:")
        for i, batch in enumerate(batches):
            print(f"  Batch {i}: indices {batch} (size={len(batch)})")
        
        # Verify first batch is [0, 2, 4, 6] (every 2nd sample starting from 0)
        # With global_batch=8 and dp_size=2, micro_batch=4
        # DP rank 0 gets indices [0, 2, 4, 6] from the first 8 samples
        expected_first_batch = [0, 2, 4, 6]
        assert batches[0] == expected_first_batch, \
            f"Expected first batch {expected_first_batch}, got {batches[0]}"
        
        print("✓ DP sampler generates correct indices for DP rank 0")
    
    dist.barrier()


def test_get_module_dp_info_for_rank():
    """Test getting DP info for current rank"""
    print("\n" + "="*60)
    print("Test get_module_dp_info_for_rank")
    print("="*60)
    
    # Create grids
    encoder_grid = create_hypercomm_grid(offset=0, tp=1, cp=1, pp=1, dp=2)
    llm_grid = create_hypercomm_grid(offset=2, tp=1, cp=1, pp=1, dp=1)
    
    module_to_grid_map = {
        'images': encoder_grid,
        'language_module': llm_grid,
    }
    
    base_batch_size = 8
    rank = dist.get_rank()
    
    # Get DP info for current rank
    dp_info = get_module_dp_info_for_rank(
        module_to_grid_map,
        base_batch_size,
        current_rank=rank
    )
    
    if rank < 2:
        # Ranks 0-1 should be in encoder with micro_batch=4
        assert dp_info is not None, f"Rank {rank} should have DP info"
        assert dp_info.micro_batch_size == 4, \
            f"Rank {rank} (encoder) should have micro_batch=4, got {dp_info.micro_batch_size}"
        print(f"✓ Rank {rank} (encoder): micro_batch={dp_info.micro_batch_size}, dp_rank={dp_info.dp_rank}")
    
    elif rank == 2:
        # Rank 2 should be in LLM with micro_batch=8
        assert dp_info is not None, f"Rank {rank} should have DP info"
        assert dp_info.micro_batch_size == 8, \
            f"Rank {rank} (llm) should have micro_batch=8, got {dp_info.micro_batch_size}"
        print(f"✓ Rank {rank} (llm): micro_batch={dp_info.micro_batch_size}, dp_rank={dp_info.dp_rank}")
    
    dist.barrier()


if __name__ == "__main__":
    # Initialize distributed training
    Utils.initialize_distributed()
    test_scenario_1()
    test_scenario_2()
    test_dp_sampler()
    test_get_module_dp_info_for_rank()

    dist.destroy_process_group()

    # try:
    #     test_scenario_1()
    #     test_scenario_2()
    #     test_dp_sampler()
    #     test_get_module_dp_info_for_rank()
        
    #     if dist.get_rank() == 0:
    #         print("\n" + "="*60)
    #         print("All tests passed! ✓")
    #         print("="*60)
    
    # finally:
    #     dist.destroy_process_group()
