import torch.distributed as dist
import torch
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_with_transformer_engine_spec
from megatron.core.models.gpt.gpt_model import GPTModel
from megatron.core.models.mimo.config.base_configs import ColocatedCommConfig, MimoModelConfig
from megatron.core.models.mimo.model.base import MimoModel
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.transformer.transformer_block import TransformerBlock
from megatron.core.transformer.spec_utils import ModuleSpec
from megatron.core.models.vision.multimodal_projector import MultimodalProjector
from examples.mimo.configs.llava_vlm import get_llava_projection_layer_spec, get_llava_projection_config
from megatron.core.models.mimo.submodules.vision import VisionModalitySubmodules
from megatron.core.distributed import DistributedDataParallel, DistributedDataParallelConfig
from tests.unit_tests.pipeline_parallel.test_multimodule_schedules import create_hypercomm_grid, _get_pg_collection_with_embedding_groups
from tests.unit_tests.models.heterogenous_parallel.config import ModelConfig
from typing import Optional

def get_language_model_spec(num_layers, hidden_size, num_attention_heads, vocab_size, seq_len, pg_collection):
    """Get the language model spec."""
    # Determine pre_process and post_process based on PP rank
    pp_rank = dist.get_rank(pg_collection.pp)
    pp_size = dist.get_world_size(pg_collection.pp)
    pre_process = (pp_rank == 0)
    post_process = (pp_rank == pp_size - 1)
    
    print(f"[get_language_model_spec] Rank {dist.get_rank()}: PP rank={pp_rank}/{pp_size}, "
          f"pre_process={pre_process}, post_process={post_process}")

    tp_size = pg_collection.tp.size() if pg_collection.tp is not None else 1
    pp_size = pg_collection.pp.size() if pg_collection.pp is not None else 1
    
    lm_config = TransformerConfig(
        num_layers=num_layers, hidden_size=hidden_size, num_attention_heads=num_attention_heads, use_cpu_initialization=True, variable_seq_lengths=True, moe_token_dispatcher_type= 'alltoall', tensor_model_parallel_size=tp_size, pipeline_model_parallel_size=pp_size, pipeline_dtype=torch.bfloat16, bf16=True,
        cross_entropy_loss_fusion=True,
        cross_entropy_fusion_impl='te',
    )
    language_layer_spec = get_gpt_layer_with_transformer_engine_spec()
    language_model_spec = ModuleSpec(
        module=GPTModel,
        params={
            "config": lm_config,
            "transformer_layer_spec": language_layer_spec,
            "vocab_size": vocab_size,
            "max_sequence_length": seq_len,
            "pre_process": pre_process,
            "post_process": post_process,
            "pg_collection": pg_collection,
        },
    )
    return language_model_spec


def get_vision_submodules_spec(num_layers, hidden_size, num_attention_heads, language_hidden_size, pg_collection):
    """Get the submodule spec for the vision modality.
    
    Args:
        num_layers: Number of transformer layers in vision encoder
        hidden_size: Hidden size of vision encoder
        num_attention_heads: Number of attention heads in vision encoder
        language_hidden_size: Hidden size of language model (for projection output)
        pg_collection: Process group collection
    """
    vision_layer_spec = get_gpt_layer_with_transformer_engine_spec()

    tp_size = pg_collection.tp.size() if pg_collection.tp is not None else 1
    pp_size = pg_collection.pp.size() if pg_collection.pp is not None else 1

    vision_config = TransformerConfig(
        num_layers=num_layers, hidden_size=hidden_size, num_attention_heads=num_attention_heads, use_cpu_initialization=True, variable_seq_lengths=True, moe_token_dispatcher_type= 'alltoall', tensor_model_parallel_size=tp_size, pipeline_model_parallel_size=pp_size, pipeline_dtype=torch.bfloat16, bf16=True,
    )
    vision_encoder_spec = ModuleSpec(
        module=TransformerBlock,
        params={
            "config": vision_config,
            "spec": vision_layer_spec,
            "pg_collection": pg_collection,
            "pre_process": True,
            "post_process": True
        },
    )

    # Create vision projection spec - projects from vision hidden size to language hidden size
    vision_projection_spec = ModuleSpec(
        module=MultimodalProjector,
        params={
            "config": get_llava_projection_config(
                hidden_size=language_hidden_size  # Output size should match language model
            ),
            "submodules": get_llava_projection_layer_spec().submodules,
            "projector_type": "mlp",
            "input_size": vision_config.hidden_size,  # Input size from vision encoder
            "tp_group": pg_collection.tp,
        },
    )

    # Create vision modality spec
    vision_submodule_spec = ModuleSpec(
        module=VisionModalitySubmodules,
        submodules={
            "encoders": {"clip_encoder": vision_encoder_spec},
            "input_projections": [vision_projection_spec],
        },
    )

    return vision_submodule_spec


def get_vlm_mimo_model(
    model_config: ModelConfig,
    seq_len: int,
):
    """Create VLM MIMO model from centralized config.
    
    Args:
        model_config: Model configuration (architectures, parallelisms, special tokens)
        seq_len: Sequence length for the language model
        
    Returns:
        Tuple of (mimo_model, module_to_grid_map, topology)
    """
    # Extract configurations
    vision_arch = model_config.get_arch(model_config.encoder_module_name)
    llm_arch = model_config.get_arch(model_config.llm_module_name)
    vision_parallel = model_config.get_parallelism(model_config.encoder_module_name)
    llm_parallel = model_config.get_parallelism(model_config.llm_module_name)
    
    # Use llm_rank_offset from config for explicit colocation control
    # offset=0: Both modules share same GPUs (colocated)
    # offset>0: Modules use different GPUs starting at different ranks
    llm_offset = model_config.llm_rank_offset
    
    language_module_grid = create_hypercomm_grid(
        offset=llm_offset, 
        tp=llm_parallel.tensor_parallel, 
        cp=1, 
        pp=llm_parallel.pipeline_parallel, 
        dp=llm_parallel.data_parallel
    )
    language_pg_collection = _get_pg_collection_with_embedding_groups(language_module_grid)

    vision_module_grid = create_hypercomm_grid(
        offset=0,  # Vision always starts at rank 0
        tp=vision_parallel.tensor_parallel, 
        cp=1, 
        pp=vision_parallel.pipeline_parallel, 
        dp=vision_parallel.data_parallel
    )
    vision_pg_collection = _get_pg_collection_with_embedding_groups(vision_module_grid)

    language_model_spec = get_language_model_spec(
        llm_arch.num_layers, 
        llm_arch.hidden_size, 
        llm_arch.num_attention_heads,
        llm_arch.vocab_size, 
        seq_len, 
        language_pg_collection
    )
    vision_submodule_spec = get_vision_submodules_spec(
        vision_arch.num_layers, 
        vision_arch.hidden_size, 
        vision_arch.num_attention_heads,
        llm_arch.hidden_size, 
        vision_pg_collection
    )

    mimo_config = MimoModelConfig(
        language_model_spec=language_model_spec,
        modality_submodules_spec={model_config.encoder_module_name: vision_submodule_spec},
        special_token_ids=model_config.special_token_ids,
    )
    # Create MIMO model
    mimo_model = MimoModel(mimo_config)
    module_to_grid_map = {
        model_config.encoder_module_name: vision_module_grid, 
        model_config.llm_module_name: language_module_grid
    }
    topology = {
        model_config.encoder_module_name: [model_config.llm_module_name],  # encoder sends to LLM
        model_config.llm_module_name: [],  # LLM is the last stage
    }


    mimo_model.to(torch.device("cuda")).to(torch.bfloat16)
    
    ddp_config = DistributedDataParallelConfig(overlap_grad_reduce=True, bucket_size=10000)
    if mimo_model.language_model is not None:
        mimo_model.language_model = DistributedDataParallel(
        config=mimo_model.language_model.config,
        ddp_config=ddp_config,
        module=mimo_model.language_model,
        pg_collection=language_pg_collection
        )
    submodule = mimo_model.modality_submodules[model_config.encoder_module_name]

    if submodule is not None:
        submodule = DistributedDataParallel(
            config=submodule.encoders['clip_encoder'].config,
            ddp_config=ddp_config,
            module=submodule,
            pg_collection=vision_pg_collection
        )
    mimo_model.modality_submodules[model_config.encoder_module_name] = submodule

    return mimo_model, module_to_grid_map, topology


def get_vlm_mimo_model_homogeneous(
    model_config: ModelConfig,
    seq_len: int,
):
    """Create VLM MIMO model for homogeneous parallelism (Case 1).
    
    This function is optimized for homogeneous parallelism where:
    - Both vision and LLM use the same parallelism strategy
    - Both modules share the same grid and process groups
    - Only one grid and pg_collection are created
    
    Args:
        model_config: Model configuration (architectures, parallelisms, special tokens)
        seq_len: Sequence length for the language model
        
    Returns:
        Tuple of (mimo_model, shared_grid, shared_pg_collection, topology)
    """
    # Validate homogeneous parallelism
    vision_parallel = model_config.get_parallelism(model_config.encoder_module_name)
    llm_parallel = model_config.get_parallelism(model_config.llm_module_name)
    
    if (vision_parallel.tensor_parallel != llm_parallel.tensor_parallel or
        vision_parallel.pipeline_parallel != llm_parallel.pipeline_parallel or
        vision_parallel.data_parallel != llm_parallel.data_parallel):
        raise ValueError(
            f"Homogeneous parallelism requires identical TP/PP/DP for all modules. "
            f"Vision: TP={vision_parallel.tensor_parallel}, PP={vision_parallel.pipeline_parallel}, DP={vision_parallel.data_parallel}. "
            f"LLM: TP={llm_parallel.tensor_parallel}, PP={llm_parallel.pipeline_parallel}, DP={llm_parallel.data_parallel}."
        )
    
    # Extract configurations
    vision_arch = model_config.get_arch(model_config.encoder_module_name)
    llm_arch = model_config.get_arch(model_config.llm_module_name)
    
    # Create single shared grid for both modules
    shared_grid = create_hypercomm_grid(
        offset=0,
        tp=llm_parallel.tensor_parallel,
        cp=1,
        pp=llm_parallel.pipeline_parallel,
        dp=llm_parallel.data_parallel
    )
    shared_pg_collection = _get_pg_collection_with_embedding_groups(shared_grid)
    
    # Create model specs using shared pg_collection
    language_model_spec = get_language_model_spec(
        llm_arch.num_layers,
        llm_arch.hidden_size,
        llm_arch.num_attention_heads,
        llm_arch.vocab_size,
        seq_len,
        shared_pg_collection
    )
    vision_submodule_spec = get_vision_submodules_spec(
        vision_arch.num_layers,
        vision_arch.hidden_size,
        vision_arch.num_attention_heads,
        llm_arch.hidden_size,
        shared_pg_collection
    )
    
    mimo_config = MimoModelConfig(
        language_model_spec=language_model_spec,
        modality_submodules_spec={model_config.encoder_module_name: vision_submodule_spec},
        special_token_ids=model_config.special_token_ids,
    )
    
    # Create MIMO model
    mimo_model = MimoModel(mimo_config)
    
    # Topology (same as heterogeneous case)
    topology = {
        model_config.encoder_module_name: [model_config.llm_module_name],  # encoder sends to LLM
        model_config.llm_module_name: [],  # LLM is the last stage
    }
    
    # Move to device
    mimo_model.to(torch.device("cuda")).to(torch.bfloat16)
    
    # Wrap both modules with DDP using shared pg_collection
    ddp_config = DistributedDataParallelConfig(overlap_grad_reduce=True, bucket_size=10000)
    
    if mimo_model.language_model is not None:
        mimo_model.language_model = DistributedDataParallel(
            config=mimo_model.language_model.config,
            ddp_config=ddp_config,
            module=mimo_model.language_model,
            pg_collection=shared_pg_collection
        )
    
    submodule = mimo_model.modality_submodules[model_config.encoder_module_name]
    if submodule is not None:
        submodule = DistributedDataParallel(
            config=submodule.encoders['clip_encoder'].config,
            ddp_config=ddp_config,
            module=submodule,
            pg_collection=shared_pg_collection
        )
    mimo_model.modality_submodules[model_config.encoder_module_name] = submodule
    
    return mimo_model, shared_grid, shared_pg_collection, topology


def get_vlm_mimo_model_colocated(
    model_config: ModelConfig,
    seq_len: int,
):
    """Create VLM MIMO model for colocated heterogeneous parallelism.
    
    This function is for colocated case where:
    - Encoder and LLM run on the SAME GPUs (PP=1 for both)
    - They can have DIFFERENT TP/DP configurations
    - Uses ColocatedBridgeCommunicator for data redistribution
    
    Args:
        model_config: Model configuration (architectures, parallelisms, special tokens)
        seq_len: Sequence length for the language model
        
    Returns:
        Tuple of (mimo_model, module_to_grid_map, topology)
    """
    # Validate colocated requirements
    vision_parallel = model_config.get_parallelism(model_config.encoder_module_name)
    llm_parallel = model_config.get_parallelism(model_config.llm_module_name)
    
    if vision_parallel.pipeline_parallel != 1 or llm_parallel.pipeline_parallel != 1:
        raise ValueError(
            f"Colocated parallelism requires PP=1 for all modules. "
            f"Vision PP={vision_parallel.pipeline_parallel}, LLM PP={llm_parallel.pipeline_parallel}."
        )
    
    if vision_parallel.total_ranks != llm_parallel.total_ranks:
        raise ValueError(
            f"Colocated parallelism requires same total ranks for all modules. "
            f"Vision: {vision_parallel.total_ranks}, LLM: {llm_parallel.total_ranks}."
        )
    
    # Extract configurations
    vision_arch = model_config.get_arch(model_config.encoder_module_name)
    llm_arch = model_config.get_arch(model_config.llm_module_name)
    
    # Create grids - both start at offset 0 (colocated)
    vision_module_grid = create_hypercomm_grid(
        offset=0,
        tp=vision_parallel.tensor_parallel,
        cp=1,
        pp=1,
        dp=vision_parallel.data_parallel
    )
    vision_pg_collection = _get_pg_collection_with_embedding_groups(vision_module_grid)
    
    language_module_grid = create_hypercomm_grid(
        offset=0,  # Same offset = colocated
        tp=llm_parallel.tensor_parallel,
        cp=1,
        pp=1,
        dp=llm_parallel.data_parallel
    )
    language_pg_collection = _get_pg_collection_with_embedding_groups(language_module_grid)
    
    # Create model specs
    language_model_spec = get_language_model_spec(
        llm_arch.num_layers,
        llm_arch.hidden_size,
        llm_arch.num_attention_heads,
        llm_arch.vocab_size,
        seq_len,
        language_pg_collection
    )
    vision_submodule_spec = get_vision_submodules_spec(
        vision_arch.num_layers,
        vision_arch.hidden_size,
        vision_arch.num_attention_heads,
        llm_arch.hidden_size,
        vision_pg_collection
    )
    
    # Build module_to_grid_map and topology
    module_to_grid_map = {
        model_config.encoder_module_name: vision_module_grid,
        model_config.llm_module_name: language_module_grid
    }
    topology = {
        model_config.encoder_module_name: [model_config.llm_module_name],
        model_config.llm_module_name: [],
    }
    
    # Create colocated communication config
    colocated_comm_config = ColocatedCommConfig(
        module_to_grid_map=module_to_grid_map,
        topology=topology,
        dim_mapping={'b': 0, 's': 1, 'h': 2},
    )
    
    # Create MIMO config with colocated communication
    mimo_config = MimoModelConfig(
        language_model_spec=language_model_spec,
        modality_submodules_spec={model_config.encoder_module_name: vision_submodule_spec},
        special_token_ids=model_config.special_token_ids,
        colocated_comm_config=colocated_comm_config,
    )
    
    # Create MIMO model
    mimo_model = MimoModel(mimo_config)
    
    # Move to device
    mimo_model.to(torch.device("cuda")).to(torch.bfloat16)
    
    # Wrap modules with DDP using their respective pg_collections
    ddp_config = DistributedDataParallelConfig(overlap_grad_reduce=True, bucket_size=10000)
    
    if mimo_model.language_model is not None:
        mimo_model.language_model = DistributedDataParallel(
            config=mimo_model.language_model.config,
            ddp_config=ddp_config,
            module=mimo_model.language_model,
            pg_collection=language_pg_collection
        )
    
    submodule = mimo_model.modality_submodules[model_config.encoder_module_name]
    if submodule is not None:
        submodule = DistributedDataParallel(
            config=submodule.encoders['clip_encoder'].config,
            ddp_config=ddp_config,
            module=submodule,
            pg_collection=vision_pg_collection
        )
    mimo_model.modality_submodules[model_config.encoder_module_name] = submodule
    
    return mimo_model, module_to_grid_map, topology

