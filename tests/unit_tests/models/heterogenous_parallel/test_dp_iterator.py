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
    validate_heterogeneous_batch_config,
    HeterogeneousDPBatchSampler,
)
from tests.unit_tests.models.heterogenous_parallel.config import (
    DataConfig,
    ModelConfig,
    ModuleArchConfig,
    ModuleParallelismConfig,
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
    
    # Create model config
    model_config = ModelConfig(
        module_architectures={
            'images': ModuleArchConfig(num_layers=4, hidden_size=768, num_attention_heads=4, seq_length=256, vocab_size=0),
            'language_module': ModuleArchConfig(num_layers=4, hidden_size=768, num_attention_heads=4, seq_length=1024, vocab_size=4000),
        },
        module_parallelisms={
            'images': ModuleParallelismConfig(tensor_parallel=2, pipeline_parallel=1, data_parallel=2),
            'language_module': ModuleParallelismConfig(tensor_parallel=2, pipeline_parallel=2, data_parallel=1),
        },
        special_token_ids={'images': 32000},
    )
    
    # Create data config
    data_config = DataConfig(
        base_batch_size=base_batch_size,
        num_microbatches=num_microbatches,
        seq_length=1024,
        image_seq_length=256,
        image_special_token_id=32000,
        vocab_size=4000,
    )
    
    # Validate configuration
    global_batch_size = validate_heterogeneous_batch_config(
        model_config=model_config,
        data_config=data_config,
    )
    
    rank = dist.get_rank()
    
    if rank == 0:
        # With base_batch=8 and llm_dp=1, global_batch per microbatch = 8
        assert global_batch_size == 8, f"Expected global_batch=8, got {global_batch_size}"
        
        # Encoder should have micro_batch=4 (global_batch=8 / encoder_dp=2)
        encoder_dp_size = model_config.get_parallelism('images').data_parallel
        encoder_micro_batch = global_batch_size // encoder_dp_size
        assert encoder_micro_batch == 4, f"Expected encoder micro_batch=4, got {encoder_micro_batch}"
        
        # LLM should have micro_batch=8 (global_batch=8 / llm_dp=1)
        llm_dp_size = model_config.get_parallelism('language_module').data_parallel
        llm_micro_batch = global_batch_size // llm_dp_size
        assert llm_micro_batch == 8, f"Expected llm micro_batch=8, got {llm_micro_batch}"
        
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
    
    # Create model config
    model_config = ModelConfig(
        module_architectures={
            'images': ModuleArchConfig(num_layers=4, hidden_size=768, num_attention_heads=4, seq_length=256, vocab_size=0),
            'language_module': ModuleArchConfig(num_layers=4, hidden_size=768, num_attention_heads=4, seq_length=1024, vocab_size=4000),
        },
        module_parallelisms={
            'images': ModuleParallelismConfig(tensor_parallel=1, pipeline_parallel=1, data_parallel=1),
            'language_module': ModuleParallelismConfig(tensor_parallel=1, pipeline_parallel=1, data_parallel=4),
        },
        special_token_ids={'images': 32000},
    )
    
    # Create data config
    data_config = DataConfig(
        base_batch_size=base_batch_size,
        num_microbatches=num_microbatches,
        seq_length=1024,
        image_seq_length=256,
        image_special_token_id=32000,
        vocab_size=4000,
    )
    
    # Validate configuration
    global_batch_size = validate_heterogeneous_batch_config(
        model_config=model_config,
        data_config=data_config,
    )
    
    rank = dist.get_rank()
    
    if rank == 0:
        # With base_batch=16 and llm_dp=4, global_batch per microbatch = 64
        assert global_batch_size == 64, f"Expected global_batch=64, got {global_batch_size}"
        
        # Encoder should have micro_batch=64 (global_batch=64 / encoder_dp=1)
        encoder_dp_size = model_config.get_parallelism('images').data_parallel
        encoder_micro_batch = global_batch_size // encoder_dp_size
        assert encoder_micro_batch == 64, f"Expected encoder micro_batch=64, got {encoder_micro_batch}"
        
        # LLM should have micro_batch=16 (global_batch=64 / llm_dp=4)
        llm_dp_size = model_config.get_parallelism('language_module').data_parallel
        llm_micro_batch = global_batch_size // llm_dp_size
        assert llm_micro_batch == 16, f"Expected llm micro_batch=16, got {llm_micro_batch}"
        
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


if __name__ == "__main__":
    # Initialize distributed training
    Utils.initialize_distributed()
    test_scenario_1()
    test_scenario_2()
    test_dp_sampler()

    if dist.get_rank() == 0:
        print("\n" + "="*60)
        print("All tests passed! ✓")
        print("="*60)
    
    dist.destroy_process_group()
