"""
Configuration classes for heterogeneous multi-module (MIMO) training.
"""

from dataclasses import dataclass, field
from typing import Dict, Optional


@dataclass(frozen=True)
class ModuleArchConfig:
    """Architecture configuration for a single module (encoder/LLM)."""
    num_layers: int
    hidden_size: int
    num_attention_heads: int
    seq_length: int
    vocab_size: int = 0  # 0 for vision encoders



@dataclass(frozen=True)
class ModuleParallelismConfig:
    """Parallelism configuration for a single module."""
    tensor_parallel: int = 1
    pipeline_parallel: int = 1
    data_parallel: int = 1
    context_parallel: int = 1
    expert_parallel: int = 1
    expert_tensor_parallel: int = 1
    
    @property
    def total_ranks(self) -> int:
        return self.tensor_parallel * self.pipeline_parallel * self.data_parallel


@dataclass(frozen=True)
class ModelConfig:
    """Complete model configuration for MIMO heterogeneous training."""
    module_architectures: Dict[str, ModuleArchConfig]
    module_parallelisms: Dict[str, ModuleParallelismConfig]
    special_token_ids: Dict[str, int] = field(default_factory=dict)
    llm_module_name: str = "language_module"
    encoder_module_name: str = "images"  # Primary encoder module name
    
    def __post_init__(self):
        if self.llm_module_name not in self.module_architectures:
            raise ValueError(f"LLM module '{self.llm_module_name}' not in architectures")
        
        if self.encoder_module_name not in self.module_architectures:
            raise ValueError(f"Encoder module '{self.encoder_module_name}' not in architectures")
        
        if set(self.module_architectures.keys()) != set(self.module_parallelisms.keys()):
            raise ValueError("Module names must match between architectures and parallelisms")
    
    @property
    def total_world_size(self) -> int:
        """Total number of ranks needed."""
        return sum(p.total_ranks for p in self.module_parallelisms.values())
    
    @property
    def module_names(self):
        return list(self.module_architectures.keys())
    
    def get_arch(self, module_name: str) -> ModuleArchConfig:
        return self.module_architectures[module_name]
    
    def get_parallelism(self, module_name: str) -> ModuleParallelismConfig:
        return self.module_parallelisms[module_name]


@dataclass(frozen=True)
class DataConfig:
    """Data configuration for heterogeneous training."""
    base_batch_size: int  # LLM micro-batch per DP replica
    num_microbatches: int
    
    # Sequence parameters
    seq_length: int  # LLM sequence length
    image_seq_length: int
    
    # Dataset parameters
    vocab_size: int
    image_special_token_id: int
    dataset_size: int = 256
    num_workers: int = 8
    prefetch_factor: int = 4
    pin_memory: bool = True
    persistent_workers: bool = True


@dataclass(frozen=True)
class RuntimeConfig:
    """Runtime configuration for training."""
    num_iterations: int
    warmup_iterations: int = 2
    log_interval: int = 1
    enable_performance_monitoring: bool = True
    metrics_output_dir: str = "./metrics"
    
    # Profiling
    enable_profiling: bool = False
    use_pytorch_profiler: bool = False
    profile_start_step: Optional[int] = None
    profile_end_step: Optional[int] = None
    tensorboard_dir: str = "./tb_logs"
    
    def __post_init__(self):
        if self.num_iterations < 1:
            raise ValueError("num_iterations must be >= 1")

