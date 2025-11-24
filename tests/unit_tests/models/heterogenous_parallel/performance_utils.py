"""
Performance monitoring for heterogeneous multi-module (MIMO) training.
"""

from datetime import datetime
from typing import Dict, Optional, Any
import logging
from types import SimpleNamespace

import torch
import torch.distributed as dist
import numpy as np

from megatron.core.timers import Timers
from megatron.training.training import num_floating_point_operations
from tests.unit_tests.models.heterogenous_parallel.config import ModelConfig, DataConfig, RuntimeConfig
import json

def create_mock_args(
    seq_length: int,
    hidden_size: int,
    num_layers: int,
    num_attention_heads: int,
    vocab_size: int,
    ffn_hidden_size: Optional[int] = None,
    kv_channels: Optional[int] = None,
) -> SimpleNamespace:
    """Create a mock args object for num_floating_point_operations function.
    
    Sets defaults for standard transformer (no MoE, GQA, MLA, hybrid models).
    
    Args:
        seq_length: Sequence length
        hidden_size: Hidden size
        num_layers: Number of layers
        num_attention_heads: Number of attention heads
        vocab_size: Vocabulary size
        ffn_hidden_size: FFN hidden size (default: 4 * hidden_size)
        kv_channels: Key-value channels (default: hidden_size / num_attention_heads)
    
    Returns:
        SimpleNamespace with all required fields for FLOPs calculation
    """
    if ffn_hidden_size is None:
        ffn_hidden_size = 4 * hidden_size
    if kv_channels is None:
        kv_channels = hidden_size // num_attention_heads
    
    args = SimpleNamespace()
    
    # Required fields for standard transformer
    args.seq_length = seq_length
    args.hidden_size = hidden_size
    args.num_layers = num_layers
    args.num_attention_heads = num_attention_heads
    args.kv_channels = kv_channels
    args.ffn_hidden_size = ffn_hidden_size
    args.padded_vocab_size = vocab_size
    args.swiglu = False
    
    # GQA settings (disabled for standard transformer)
    args.group_query_attention = False
    args.num_query_groups = num_attention_heads
    
    # MoE settings (disabled)
    args.num_experts = None
    args.moe_layer_freq = None
    args.moe_ffn_hidden_size = None
    args.moe_shared_expert_intermediate_size = None
    args.moe_router_topk = None
    
    # MTP settings (disabled)
    args.mtp_num_layers = None
    
    # MLA settings (disabled)
    args.multi_latent_attention = False
    args.q_lora_rank = None
    args.kv_lora_rank = None
    args.qk_head_dim = None
    args.qk_pos_emb_head_dim = None
    args.v_head_dim = None
    
    # Hybrid model settings (disabled)
    args.is_hybrid_model = False
    args.hybrid_override_pattern = None
    args.hybrid_attention_ratio = 1.0
    args.hybrid_mlp_ratio = 0.0
    args.mamba_state_dim = None
    args.mamba_head_dim = None
    args.mamba_num_groups = None
    args.mamba_num_heads = None
    
    return args


class PerformanceMonitor:
    """Performance monitor for heterogeneous multi-module (MIMO) training.
    
    Uses centralized configs to calculate TFLOPs/s/GPU, samples/sec, and tokens/sec metrics.
    """
    
    def __init__(
        self,
        model_config: ModelConfig,
        data_config: DataConfig,
        runtime_config: RuntimeConfig,
    ):
        """Initialize performance monitor from centralized configs.
        
        Args:
            model_config: Model configuration (architectures, parallelisms)
            data_config: Data configuration (batch sizes)
            runtime_config: Runtime configuration (warmup, logging)
        """
        self.model_config = model_config
        self.data_config = data_config
        self.runtime_config = runtime_config
        
        # Calculate parameters from architecture
        self.vision_params = self._calculate_params_from_arch(
            model_config.get_arch(model_config.encoder_module_name)
        )
        self.llm_params = self._calculate_params_from_arch(
            model_config.get_arch(model_config.llm_module_name)
        )
        
        # Get rank and world size
        self.rank = dist.get_rank() if dist.is_initialized() else 0
        self.world_size = dist.get_world_size() if dist.is_initialized() else 1
        
        # Calculate global batch size from LLM's DP configuration
        llm_parallelism = model_config.get_parallelism(model_config.llm_module_name)
        self.global_batch_size = data_config.base_batch_size * llm_parallelism.data_parallel
        
        # Calculate total system FLOPs per iteration
        self.total_system_flops = self._calculate_total_flops()
        
        # Get LLM sequence length for token throughput calculation
        llm_arch = model_config.get_arch(model_config.llm_module_name)
        self.llm_seq_length = llm_arch.seq_length
        
        # Calculate effective global batch (total samples per iteration)
        self.effective_global_batch = self.global_batch_size * data_config.num_microbatches
        
        # Create timers
        self.timers = Timers(log_level=1, log_option='minmax')
        
        # Per-iteration history
        self.history = []
        
        # Log configuration on rank 0
        if self.rank == 0:
            self._log_configuration()
    
    def _calculate_params_from_arch(self, arch_config) -> int:
        """Calculate total parameters from architecture config.
        
        Following megatron/training/theoretical_memory_usage.py approach.
        
        Args:
            arch_config: ModuleArchConfig with hidden_size, num_layers, vocab_size
            
        Returns:
            Total parameter count
        """
        h = arch_config.hidden_size
        n = arch_config.num_layers
        v = arch_config.vocab_size
        
        # Per-layer: attention (4h^2) + FFN (8h^2 for 4h FFN) + layernorms (4h)
        params_per_layer = 12 * h * h + 4 * h
        transformer_params = params_per_layer * n
        
        # Embeddings
        embedding_params = h * v if v > 0 else 0
        
        return transformer_params + embedding_params
    
    def _calculate_total_flops(self) -> int:
        """Calculate total FLOPs per iteration across all modules.
        
        Uses global_batch_size which is the same for all modules.
        
        Returns:
            Total FLOPs per iteration (sum across all modules and microbatches)
        """
        total_flops = 0
        
        for module_name in self.model_config.module_names:
            arch_config = self.model_config.get_arch(module_name)
            
            # Create mock args for FLOPs calculation
            module_args = create_mock_args(
                seq_length=arch_config.seq_length,
                hidden_size=arch_config.hidden_size,
                num_layers=arch_config.num_layers,
                num_attention_heads=arch_config.num_attention_heads,
                vocab_size=arch_config.vocab_size,
            )
            
            # FLOPs for one microbatch using global batch size
            # All modules process the same global_batch_size
            flops_per_microbatch = num_floating_point_operations(
                module_args, 
                self.global_batch_size
            )
            
            # Total FLOPs for this module across all microbatches
            module_total_flops = flops_per_microbatch * self.data_config.num_microbatches
            
            total_flops += module_total_flops
            
            if self.rank == 0:
                logging.debug(
                    f"Module {module_name}: "
                    f"{flops_per_microbatch:.2e} FLOPs/microbatch × "
                    f"{self.data_config.num_microbatches} microbatches = "
                    f"{module_total_flops:.2e} FLOPs total"
                )
        
        return total_flops
    
    def _log_configuration(self):
        """Log the performance monitoring configuration."""
        logging.info("=" * 80)
        logging.info("Performance Monitor Configuration")
        logging.info("=" * 80)
        logging.info(f"Global batch size per microbatch: {self.global_batch_size}")
        logging.info(f"Number of microbatches: {self.data_config.num_microbatches}")
        logging.info(f"Effective global batch per iteration: {self.effective_global_batch}")
        logging.info(f"Total GPUs (world size): {self.world_size}")
        logging.info(f"Total system FLOPs per iteration: {self.total_system_flops:.2e}")
        logging.info("-" * 80)
        logging.info(f"{'Module':<20} | {'SeqLen':>6} | {'Hidden':>6} | {'Layers':>6} | {'Heads':>5}")
        logging.info("-" * 80)
        
        for module_name in self.model_config.module_names:
            arch_config = self.model_config.get_arch(module_name)
            logging.info(
                f"{module_name:<20} | "
                f"{arch_config.seq_length:>6} | "
                f"{arch_config.hidden_size:>6} | "
                f"{arch_config.num_layers:>6} | "
                f"{arch_config.num_attention_heads:>5}"
            )
        
        logging.info("=" * 80)
    
    def start_iteration(self):
        """Start timing an iteration."""
        self.timers('iteration-time', log_level=0).start(barrier=False)
    
    def end_iteration(self):
        """End timing and record metrics for this iteration."""
        self.timers('iteration-time').stop()
        
        # Get per-iteration min/max across ranks
        iter_stats = self._get_minmax_across_ranks('iteration-time')
        fb_stats = self._get_minmax_across_ranks('forward-backward')
        
        # Collect memory usage from all ranks
        memory_stats = self._get_memory_stats_all_ranks()
        
        # Use max time as the bottleneck
        iter_time_sec = iter_stats['max'] / 1000.0 if iter_stats else 0
        
        if iter_time_sec > 0:
            # Samples per second (global batch processed in one iteration)
            samples_per_sec = self.effective_global_batch / iter_time_sec
            
            # Tokens per second (for LLM output)
            tokens_per_sec = (self.effective_global_batch * self.llm_seq_length) / iter_time_sec
            
            # TFLOPs per GPU
            # Total work divided by (time × number of GPUs)
            tflops_per_gpu = self.total_system_flops / (iter_time_sec * 1e12 * self.world_size)
        else:
            samples_per_sec = 0
            tokens_per_sec = 0
            tflops_per_gpu = 0
        
        # Store metrics
        self.history.append({
            'iteration': len(self.history) + 1,
            'iteration_time_min_ms': iter_stats['min'] if iter_stats else 0,
            'iteration_time_max_ms': iter_stats['max'] if iter_stats else 0,
            'forward_backward_time_min_ms': fb_stats['min'] if fb_stats else 0,
            'forward_backward_time_max_ms': fb_stats['max'] if fb_stats else 0,
            'samples_per_sec': samples_per_sec,
            'tokens_per_sec': tokens_per_sec,
            'tflops_per_gpu': tflops_per_gpu,
            'memory_allocated_per_rank_gb': memory_stats['per_rank'],
            'memory_allocated_max_gb': memory_stats['max'],
        })
    
    def _get_minmax_across_ranks(self, timer_name: str) -> Optional[Dict[str, float]]:
        """Get min/max for a timer across all ranks for current iteration.
        
        Args:
            timer_name: Name of the timer to query
        
        Returns:
            Dict with 'min' and 'max' keys in milliseconds, or None if not available
        """
        if not dist.is_initialized():
            return None
        
        result = self.timers._get_global_min_max_time(
            names=[timer_name],
            reset=True,
            barrier=False,
            normalizer=0.001  # Convert to ms
        )
        
        if timer_name not in result:
            return None
        
        min_ms, max_ms = result[timer_name]
        return {'min': min_ms, 'max': max_ms}
    
    def _get_memory_stats_all_ranks(self) -> Dict[str, Any]:
        """Get memory usage statistics from all ranks.
        
        Returns:
            Dict with 'per_rank' (list of memory per rank) and 'max' (max across ranks)
        """
        if not torch.cuda.is_available():
            return {'per_rank': [], 'max': 0.0}
        
        # Get memory allocated on this rank (in GB)
        local_memory_gb = torch.cuda.max_memory_allocated() / (1024 ** 3)
        
        if not dist.is_initialized():
            return {'per_rank': [local_memory_gb], 'max': local_memory_gb}
        
        # Gather memory from all ranks
        memory_tensor = torch.tensor([local_memory_gb], dtype=torch.float32, device='cuda')
        gathered_memory = [torch.zeros_like(memory_tensor) for _ in range(self.world_size)]
        
        dist.all_gather(gathered_memory, memory_tensor)
        
        # Convert to list of floats
        per_rank_memory = [float(m.item()) for m in gathered_memory]
        max_memory = max(per_rank_memory)
        
        return {'per_rank': per_rank_memory, 'max': max_memory}
    
    def log_performance(self, iteration: int):
        """Log performance metrics for current iteration.
        
        Args:
            iteration: Current iteration number (1-indexed)
        """
        if self.runtime_config.log_interval == 0 or iteration % self.runtime_config.log_interval != 0:
            return
        
        if self.rank != 0 or not self.history:
            return
        
        metrics = self.history[-1]
        warmup_str = " (warmup)" if iteration <= self.runtime_config.warmup_iterations else ""
        
        logging.info(
            f"Iter {iteration:4d}{warmup_str} | "
            f"Time: {metrics['iteration_time_min_ms']:.1f}-{metrics['iteration_time_max_ms']:.1f} ms | "
            f"F/B: {metrics['forward_backward_time_min_ms']:.1f}-{metrics['forward_backward_time_max_ms']:.1f} ms | "
            f"TFLOPs/GPU: {metrics['tflops_per_gpu']:.1f} | "
            f"Samples/s: {metrics['samples_per_sec']:.1f} | "
            f"Tokens/s: {metrics['tokens_per_sec']:.0f} | "
            f"Mem: {metrics['memory_allocated_max_gb']:.2f} GB"
        )
    
    def save_metrics_to_file(
        self,
        filepath: str,
        extra_info: Optional[Dict[str, Any]] = None,
        exclude_warmup: bool = True,
    ):
        """Save metrics to JSON file with median statistics.
        
        Args:
            filepath: Path to save JSON file
            extra_info: Additional metadata to include
            exclude_warmup: Whether to exclude warmup iterations from statistics
        """
        if self.rank != 0:
            return
        
        history = self.history
        
        # Filter warmup
        start_idx = self.runtime_config.warmup_iterations if exclude_warmup else 0
        filtered = history[start_idx:]
        
        if not filtered:
            logging.warning("No performance data to save")
            return
        
        # Extract arrays
        iter_time_mins = np.array([h['iteration_time_min_ms'] for h in filtered])
        iter_time_maxs = np.array([h['iteration_time_max_ms'] for h in filtered])
        fb_time_mins = np.array([h['forward_backward_time_min_ms'] for h in filtered])
        fb_time_maxs = np.array([h['forward_backward_time_max_ms'] for h in filtered])
        samples_per_sec = np.array([h['samples_per_sec'] for h in filtered])
        tokens_per_sec = np.array([h['tokens_per_sec'] for h in filtered])
        tflops_per_gpu = np.array([h['tflops_per_gpu'] for h in filtered])
        memory_max_gb = np.array([h['memory_allocated_max_gb'] for h in filtered])
        
        # Get architecture info
        vision_arch = self.model_config.get_arch(self.model_config.encoder_module_name)
        llm_arch = self.model_config.get_arch(self.model_config.llm_module_name)
        vision_parallel = self.model_config.get_parallelism(self.model_config.encoder_module_name)
        llm_parallel = self.model_config.get_parallelism(self.model_config.llm_module_name)
        
        # Use calculated parameter counts (in billions)
        vision_params = self.vision_params / 1e9
        llm_params = self.llm_params / 1e9
        
        # Compute medians
        median_iter_time_sec = float(np.median(iter_time_maxs)) / 1000
        median_fb_min_ms = float(np.median(fb_time_mins))
        median_fb_max_ms = float(np.median(fb_time_maxs))
        median_tokens_per_sec = float(np.median(tokens_per_sec))
        median_tflops_per_gpu = float(np.median(tflops_per_gpu))
        max_memory_allocated_gb = float(np.max(memory_max_gb))
        
        # Build row data matching Excel columns
        row_data = {
            'Experiment': extra_info.get('exp_name', 'unknown') if extra_info else 'unknown',
            # Vision encoder
            'vision_hidden_size': vision_arch.hidden_size,
            'vision_attention_heads': vision_arch.num_attention_heads,
            'vision_total_parameters': vision_params,
            'vision_num_layers': vision_arch.num_layers,
            # Language model
            'llm_hidden_size': llm_arch.hidden_size,
            'llm_attention_heads': llm_arch.num_attention_heads,
            'llm_total_parameters': llm_params,
            'llm_num_layers': llm_arch.num_layers,
            # Data
            'vision_seq_length': self.data_config.image_seq_length,
            'total_seq_length': self.data_config.seq_length,
            'llm_microbatch_size': self.data_config.base_batch_size,
            'global_batch_size': self.effective_global_batch,
            # Vision model parallelism
            'vision_tp': vision_parallel.tensor_parallel,
            'vision_pp': vision_parallel.pipeline_parallel,
            'vision_dp': vision_parallel.data_parallel,
            # Language model parallelism
            'llm_tp': llm_parallel.tensor_parallel,
            'llm_pp': llm_parallel.pipeline_parallel,
            'llm_dp': llm_parallel.data_parallel,
            # Metrics
            'tokens_per_sec': round(median_tokens_per_sec, 1),
            'throughput_tflops_per_gpu': round(median_tflops_per_gpu, 3),
            'fwd_bwd_min_time_ms': round(median_fb_min_ms, 2),
            'fwd_bwd_max_time_ms': round(median_fb_max_ms, 2),
            'iteration_time_sec': round(median_iter_time_sec, 3),
            'max_memory_allocated_gb': round(max_memory_allocated_gb, 2),
        }
        
        # Collect per-rank memory data for JSON
        per_rank_memory_all_iters = [h['memory_allocated_per_rank_gb'] for h in filtered]
        
        # Save full metrics to JSON
        metrics = {
            'timestamp': datetime.now().isoformat(),
            'row_data': row_data,
            'detailed_stats': {
                'warmup_iterations': self.runtime_config.warmup_iterations,
                'total_iterations': len(history),
                'analyzed_iterations': len(filtered),
            },
            'memory_per_rank_gb': {
                'per_iteration': per_rank_memory_all_iters,
                'summary': {
                    'max_across_all_ranks_and_iters': max_memory_allocated_gb,
                    'per_rank_max': [float(np.max([per_rank_memory_all_iters[i][rank] 
                                                    for i in range(len(per_rank_memory_all_iters))]))
                                     for rank in range(len(per_rank_memory_all_iters[0]))] if per_rank_memory_all_iters and per_rank_memory_all_iters[0] else [],
                }
            },
        }
        
        if extra_info:
            metrics['extra_info'] = extra_info
        
        # Save JSON
        with open(filepath, 'w') as f:
            json.dump(metrics, f, indent=2)
        
        # Save CSV
        import csv
        csv_path = filepath.replace('.json', '.csv')
        with open(csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=row_data.keys())
            writer.writeheader()
            writer.writerow(row_data)
        
        # Log summary
        logging.info("=" * 80)
        logging.info(f"Performance Summary (excluding first {self.runtime_config.warmup_iterations} warmup iterations):")
        logging.info("-" * 80)
        logging.info(f"  Iteration time:   {median_iter_time_sec:.3f} s")
        logging.info(f"  Forward-backward: {median_fb_min_ms:.1f}-{median_fb_max_ms:.1f} ms")
        logging.info(f"  Throughput:")
        logging.info(f"    - {median_tflops_per_gpu:.2f} TFLOPs/s/GPU")
        logging.info(f"    - {median_tokens_per_sec:.0f} tokens/s")
        logging.info(f"  Memory:")
        logging.info(f"    - Max allocated: {max_memory_allocated_gb:.2f} GB")
        logging.info("=" * 80)
        logging.info(f"Metrics saved to: {filepath}")
        logging.info(f"CSV saved to: {csv_path}")


def create_performance_monitor(
    model_config: ModelConfig,
    data_config: DataConfig,
    runtime_config: RuntimeConfig,
    megatron_config,
) -> PerformanceMonitor:
    """Create performance monitor from centralized configs.
    
    Args:
        model_config: Model configuration (architectures, parallelisms)
        data_config: Data configuration (batch sizes)
        runtime_config: Runtime configuration (warmup, logging, profiling)
        megatron_config: Megatron config to attach timers to
    
    Returns:
        PerformanceMonitor instance with timers attached to megatron_config
    """
    monitor = PerformanceMonitor(
        model_config=model_config,
        data_config=data_config,
        runtime_config=runtime_config,
    )
    
    # Attach timers to megatron config
    megatron_config.timers = monitor.timers
    
    return monitor
