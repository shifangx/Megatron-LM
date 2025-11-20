"""Simple performance monitoring - stores per-iteration min/max, computes median."""

import time
from datetime import datetime
from typing import Dict, List, Optional, Any
import logging
from types import SimpleNamespace

import torch
import torch.distributed as dist
import numpy as np

from megatron.core.timers import Timers
from megatron.training.training import num_floating_point_operations


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
    """Simple performance monitor - stores per-iteration metrics."""
    
    def __init__(
        self,
        batch_size: int,
        num_microbatches: int,
        seq_length: int,
        all_module_configs: Dict[str, Dict[str, Any]],
        warmup_iterations: int = 2,
        log_interval: int = 1,
        enabled: bool = True,
    ):
        """Initialize performance monitor.
        
        Args:
            batch_size: Micro-batch size per training step
            num_microbatches: Number of microbatches per global batch
            seq_length: Sequence length (for output tokens, used for throughput calculation)
            all_module_configs: Dict mapping module name to config for ALL modules in the system:
                Each config has keys: seq_length, hidden_size, num_layers, num_heads, vocab_size, dp_size
            warmup_iterations: Number of initial iterations to exclude
            log_interval: Log every N iterations (0 to disable)
            enabled: Whether monitoring is enabled
        """
        self.batch_size = batch_size
        self.num_microbatches = num_microbatches
        self.seq_length = seq_length
        self.all_module_configs = all_module_configs
        self.warmup_iterations = warmup_iterations
        self.log_interval = log_interval
        self.enabled = enabled
        
        # Calculate total system FLOPs per iteration (across all modules, all microbatches)
        # Think of it globally: each microbatch flows through all modules
        self.total_system_flops = 0
        
        for module_name, module_config in all_module_configs.items():
            # FLOPs for processing one microbatch through this module
            module_args = create_mock_args(
                seq_length=module_config['seq_length'],
                hidden_size=module_config['hidden_size'],
                num_layers=module_config['num_layers'],
                num_attention_heads=module_config['num_heads'],
                vocab_size=module_config['vocab_size'],
            )
            # batch_size samples per microbatch
            flops_per_microbatch = num_floating_point_operations(module_args, batch_size)
            
            # Total for this module: num_microbatches × DP_size (data parallel replication)
            module_dp_size = module_config.get('dp_size', 1)
            module_total_flops = flops_per_microbatch * num_microbatches * module_dp_size
            
            self.total_system_flops += module_total_flops
        
        # Create timers - only level 0 and 1
        self.timers = Timers(log_level=1, log_option='minmax')
        
        # Per-iteration history
        self.history = []
        
        self.rank = dist.get_rank() if dist.is_initialized() else 0
        self.world_size = dist.get_world_size() if dist.is_initialized() else 1
        
    def start_iteration(self):
        """Start timing an iteration."""
        if not self.enabled:
            return
        self.timers('iteration-time', log_level=0).start(barrier=False)
        
    def end_iteration(self):
        """End timing and record per-iteration min/max across ranks."""
        if not self.enabled:
            return
            
        self.timers('iteration-time').stop()
        
        # Get per-iteration min/max across ranks
        iter_stats = self._get_minmax_across_ranks('iteration-time')
        fb_stats = self._get_minmax_across_ranks('forward-backward')
        
        # Calculate throughput (use max time as it's the bottleneck)
        global_batch_size = self.batch_size * self.num_microbatches * self.world_size
        iter_time_sec = iter_stats['max'] / 1000.0 if iter_stats else 0
        
        # Tokens per second
        tokens_per_sec = (global_batch_size * self.seq_length) / iter_time_sec if iter_time_sec > 0 else 0
        
        # TFLOPS per GPU (same as Megatron training.py line 1541-1543)
        # total_system_flops is total work done by entire system per iteration
        # Divide by world_size to get per-GPU throughput
        tflops_per_gpu = self.total_system_flops / (iter_time_sec * 1e12 * self.world_size) if iter_time_sec > 0 else 0
        
        # Store
        self.history.append({
            'iteration': len(self.history) + 1,
            'iteration_time_min_ms': iter_stats['min'] if iter_stats else 0,
            'iteration_time_max_ms': iter_stats['max'] if iter_stats else 0,
            'forward_backward_time_min_ms': fb_stats['min'] if fb_stats else 0,
            'forward_backward_time_max_ms': fb_stats['max'] if fb_stats else 0,
            'tokens_per_sec': tokens_per_sec,
            'tflops_per_gpu': tflops_per_gpu,
        })
    
    def _get_minmax_across_ranks(self, timer_name: str) -> Optional[Dict[str, float]]:
        """Get min/max for a timer across all ranks for current iteration."""
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
    
    def log_performance(self, iteration: int):
        """Log performance for current iteration."""
        if not self.enabled or self.log_interval == 0 or iteration % self.log_interval != 0:
            return
        
        if self.rank != 0 or not self.history:
            return
        
        metrics = self.history[-1]
        warmup_str = " (warmup)" if iteration <= self.warmup_iterations else ""
        
        logging.info(
            f"Iter {iteration:4d}{warmup_str} | "
            f"Time: {metrics['iteration_time_min_ms']:.1f}-{metrics['iteration_time_max_ms']:.1f} ms | "
            f"F/B: {metrics['forward_backward_time_min_ms']:.1f}-{metrics['forward_backward_time_max_ms']:.1f} ms | "
            f"TFLOPs/GPU: {metrics['tflops_per_gpu']:.1f} | "
            f"Tokens/s: {metrics['tokens_per_sec']:.0f}"
        )
    
    def save_metrics_to_file(
        self, 
        filepath: str,
        extra_info: Optional[Dict[str, Any]] = None,
        exclude_warmup: bool = True,
    ):
        """Save metrics - compute median of per-iteration min/max values.
        
        Only rank 0 writes to file (all ranks already have same history).
        """
        if not self.enabled or self.rank != 0:
            return
        
        import json
        
        history = self.history
        
        # Filter warmup
        start_idx = self.warmup_iterations if exclude_warmup else 0
        filtered = history[start_idx:]
        
        if not filtered:
            return
        
        # Extract arrays
        iter_time_mins = np.array([h['iteration_time_min_ms'] for h in filtered])
        iter_time_maxs = np.array([h['iteration_time_max_ms'] for h in filtered])
        fb_time_mins = np.array([h['forward_backward_time_min_ms'] for h in filtered])
        fb_time_maxs = np.array([h['forward_backward_time_max_ms'] for h in filtered])
        tokens_per_sec = np.array([h['tokens_per_sec'] for h in filtered])
        tflops_per_gpu = np.array([h['tflops_per_gpu'] for h in filtered])
        
        # Compute medians
        metrics = {
            'timestamp': datetime.now().isoformat(),
            'config': {
                'batch_size': self.batch_size,
                'num_microbatches': self.num_microbatches,
                'seq_length': self.seq_length,
                'all_module_configs': self.all_module_configs,
                'world_size': self.world_size,
                'warmup_iterations': self.warmup_iterations,
                'total_iterations': len(history),
                'analyzed_iterations': len(filtered),
            },
            'iteration_time': {
                'median_min_ms': float(np.median(iter_time_mins)),
                'median_max_ms': float(np.median(iter_time_maxs)),
            },
            'forward_backward_time': {
                'median_min_ms': float(np.median(fb_time_mins)),
                'median_max_ms': float(np.median(fb_time_maxs)),
            },
            'throughput': {
                'median_tokens_per_sec': float(np.median(tokens_per_sec)),
                'median_tflops_per_gpu': float(np.median(tflops_per_gpu)),
            },
        }
        
        if extra_info:
            metrics['extra_info'] = extra_info
        
        # Save
        with open(filepath, 'w') as f:
            json.dump(metrics, f, indent=2)
        
        # Log summary
        it = metrics['iteration_time']
        fb = metrics['forward_backward_time']
        tp = metrics['throughput']
        logging.info(f"\nPerformance Summary (excluding first {self.warmup_iterations} warmup iterations):")
        logging.info(f"  Iteration time: {it['median_min_ms']:.1f}-{it['median_max_ms']:.1f} ms (median min-max)")
        logging.info(f"  Forward-backward: {fb['median_min_ms']:.1f}-{fb['median_max_ms']:.1f} ms (median min-max)")
        logging.info(f"  Throughput: {tp['median_tflops_per_gpu']:.1f} TFLOPs/GPU, {tp['median_tokens_per_sec']:.0f} tokens/s (median)")
        logging.info(f"  Saved to: {filepath}\n")


def create_performance_monitor(
    config,
    batch_size: int,
    num_microbatches: int,
    seq_length: int,
    all_module_configs: Dict[str, Dict[str, Any]],
    warmup_iterations: int = 2,
    log_interval: int = 1,
    enabled: bool = True,
) -> PerformanceMonitor:
    """Create and attach performance monitor to config.
    
    Args:
        config: Model config to attach timers to
        batch_size: Micro-batch size
        num_microbatches: Number of microbatches
        seq_length: Sequence length (for output tokens)
        all_module_configs: Dict of ALL modules in the system, mapping name to config with:
            - seq_length, hidden_size, num_layers, num_heads, vocab_size, dp_size
        warmup_iterations: Number of warmup iterations to exclude
        log_interval: Log every N iterations (0 to disable)
        enabled: Whether monitoring is enabled
    """
    monitor = PerformanceMonitor(
        batch_size=batch_size,
        num_microbatches=num_microbatches,
        seq_length=seq_length,
        all_module_configs=all_module_configs,
        warmup_iterations=warmup_iterations,
        log_interval=log_interval,
        enabled=enabled,
    )
    
    config.timers = monitor.timers
    return monitor
