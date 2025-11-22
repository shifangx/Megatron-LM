"""
Configuration loader for experiment ablations.

Loads YAML configs and converts them to ModelConfig, DataConfig, and RuntimeConfig objects.
Supports config inheritance from baseline.yaml.
"""

import yaml
from pathlib import Path
from typing import Dict, Any
from tests.unit_tests.models.heterogenous_parallel.config import (
    ModelConfig, ModuleArchConfig, ModuleParallelismConfig,
    DataConfig, RuntimeConfig
)


def load_yaml_config(config_path: str) -> Dict[str, Any]:
    """Load YAML configuration file.
    
    Args:
        config_path: Path to YAML config file
        
    Returns:
        Dictionary containing configuration
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def deep_merge(base: Dict, override: Dict) -> Dict:
    """Deep merge two dictionaries.
    
    Args:
        base: Base dictionary
        override: Override dictionary (takes precedence)
        
    Returns:
        Merged dictionary
    """
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def load_baseline_config() -> Dict[str, Any]:
    """Load baseline configuration.
    
    Returns:
        Baseline configuration dictionary
    """
    baseline_path = Path(__file__).parent / "configs" / "baseline.yaml"
    return load_yaml_config(str(baseline_path))


def parse_model_config(config_dict: Dict[str, Any]) -> ModelConfig:
    """Parse model configuration from dictionary.
    
    Args:
        config_dict: Dictionary containing model configuration
        
    Returns:
        ModelConfig object
    """
    module_architectures = {}
    for module_name, arch_dict in config_dict['module_architectures'].items():
        module_architectures[module_name] = ModuleArchConfig(
            num_layers=arch_dict['num_layers'],
            hidden_size=arch_dict['hidden_size'],
            num_attention_heads=arch_dict['num_attention_heads'],
            seq_length=arch_dict['seq_length'],
            vocab_size=arch_dict['vocab_size'],
        )
    
    module_parallelisms = {}
    for module_name, parallel_dict in config_dict['module_parallelisms'].items():
        module_parallelisms[module_name] = ModuleParallelismConfig(
            tensor_parallel=parallel_dict['tensor_parallel'],
            pipeline_parallel=parallel_dict['pipeline_parallel'],
            data_parallel=parallel_dict['data_parallel'],
        )
    
    return ModelConfig(
        module_architectures=module_architectures,
        module_parallelisms=module_parallelisms,
        special_token_ids=config_dict['special_token_ids'],
        llm_module_name=config_dict['llm_module_name'],
        encoder_module_name=config_dict['encoder_module_name'],
    )


def parse_data_config(config_dict: Dict[str, Any]) -> DataConfig:
    """Parse data configuration from dictionary.
    
    Args:
        config_dict: Dictionary containing data configuration
        
    Returns:
        DataConfig object
    """
    return DataConfig(
        base_batch_size=config_dict['base_batch_size'],
        num_microbatches=config_dict['num_microbatches'],
        seq_length=config_dict['seq_length'],
        image_seq_length=config_dict['image_seq_length'],
        vocab_size=config_dict['vocab_size'],
        image_special_token_id=config_dict['image_special_token_id'],
        dataset_size=config_dict['dataset_size'],
        num_workers=config_dict['num_workers'],
        prefetch_factor=config_dict['prefetch_factor'],
    )


def parse_runtime_config(config_dict: Dict[str, Any], experiment_dir: str = None) -> RuntimeConfig:
    """Parse runtime configuration from dictionary.
    
    Args:
        config_dict: Dictionary containing runtime configuration
        experiment_dir: Directory for experiment outputs (overrides config if provided)
        
    Returns:
        RuntimeConfig object
    """
    metrics_dir = experiment_dir if experiment_dir else config_dict['metrics_output_dir']
    tensorboard_dir = experiment_dir if experiment_dir else config_dict['tensorboard_dir']
    
    return RuntimeConfig(
        num_iterations=config_dict['num_iterations'],
        warmup_iterations=config_dict['warmup_iterations'],
        log_interval=config_dict['log_interval'],
        enable_performance_monitoring=config_dict['enable_performance_monitoring'],
        metrics_output_dir=metrics_dir,
        enable_profiling=config_dict['enable_profiling'],
        use_pytorch_profiler=config_dict['use_pytorch_profiler'],
        profile_start_step=config_dict['profile_start_step'],
        profile_end_step=config_dict['profile_end_step'],
        tensorboard_dir=tensorboard_dir,
    )


def load_experiment_config(config_path: str, experiment_dir: str = None) -> tuple:
    """Load experiment configuration with baseline merging.
    
    Args:
        config_path: Path to YAML config file (overrides baseline values)
        experiment_dir: Optional experiment directory for outputs
        
    Returns:
        Tuple of (ModelConfig, DataConfig, RuntimeConfig)
    """
    # Always start with baseline
    baseline = load_baseline_config()
    
    # Load override config and merge
    override = load_yaml_config(config_path)
    config = deep_merge(baseline, override)
    
    model_config = parse_model_config(config['model'])
    data_config = parse_data_config(config['data'])
    runtime_config = parse_runtime_config(config['runtime'], experiment_dir)
    
    return model_config, data_config, runtime_config


def generate_experiment_name(model_config: ModelConfig, data_config: DataConfig) -> str:
    """Generate descriptive experiment name from config.
    
    Args:
        model_config: Model configuration
        data_config: Data configuration
        
    Returns:
        String experiment name
    """
    vision_arch = model_config.get_arch('images')
    llm_arch = model_config.get_arch(model_config.llm_module_name)
    
    vision_parallel = model_config.get_parallelism('images')
    llm_parallel = model_config.get_parallelism(model_config.llm_module_name)
    
    name_parts = [
        f"ve_l{vision_arch.num_layers}_h{vision_arch.hidden_size}_a{vision_arch.num_attention_heads}",
        f"tp{vision_parallel.tensor_parallel}_pp{vision_parallel.pipeline_parallel}_dp{vision_parallel.data_parallel}",
        f"llm_l{llm_arch.num_layers}_h{llm_arch.hidden_size}_a{llm_arch.num_attention_heads}",
        f"tp{llm_parallel.tensor_parallel}_pp{llm_parallel.pipeline_parallel}_dp{llm_parallel.data_parallel}",
        f"mb{data_config.num_microbatches}_bs{data_config.base_batch_size}",
    ]
    
    return "_".join(name_parts)

