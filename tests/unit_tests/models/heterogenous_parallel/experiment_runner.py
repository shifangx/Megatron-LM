"""
Runs multiple experiments based on YAML configs.

uv run python -m torch.distributed.run --nproc_per_node=2 tests/unit_tests/models/heterogenous_parallel/experiment_runner.py --config tests/unit_tests/models/heterogenous_parallel/configs/baseline.yaml

"""

import os
import sys
import logging
import json
import torch
import torch.distributed as dist
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any

from tests.unit_tests.test_utilities import Utils
from tests.unit_tests.models.heterogenous_parallel.train import test_1f_1b_schedule_vlm_mimo_model_custom_pgs
from tests.unit_tests.models.heterogenous_parallel.config_loader import (
    load_experiment_config,
    generate_experiment_name
)

logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s',
    force=True
)


class ExperimentRunner:
    """Manages running multiple experiments and aggregating results."""
    
    def __init__(self, experiments_dir: str, results_dir: str = None):
        """Initialize experiment runner.
        
        Args:
            experiments_dir: Directory containing experiment YAML configs
            results_dir: Directory to save all experiment results (default: experiments_dir/results)
        """
        self.experiments_dir = Path(experiments_dir)
        self.results_dir = Path(results_dir) if results_dir else self.experiments_dir / "results"
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # Create timestamp for this run
        self.run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir = self.results_dir / f"run_{self.run_timestamp}"
        self.run_dir.mkdir(parents=True, exist_ok=True)
        
        logging.info(f"Experiment runner initialized. Results will be saved to: {self.run_dir}")
    
    def find_config_files(self) -> List[Path]:
        """Find all YAML config files in experiments directory.
        
        Returns:
            List of paths to config files
        """
        config_files = list(self.experiments_dir.glob("*.yaml")) + list(self.experiments_dir.glob("*.yml"))
        config_files = sorted(config_files)
        logging.info(f"Found {len(config_files)} config files: {[f.name for f in config_files]}")
        return config_files
    
    def run_experiment(self, config_path: Path, experiment_idx: int, total_experiments: int):
        """Run a single experiment.
        
        Args:
            config_path: Path to experiment config file
            experiment_idx: Index of this experiment (0-based)
            total_experiments: Total number of experiments
        """
        # Utils.initialize_distributed()

        rank = dist.get_rank() if dist.is_initialized() else 0
        
        logging.info(f"\n{'='*80}")
        logging.info(f"Rank {rank}: Running experiment {experiment_idx + 1}/{total_experiments}: {config_path.name}")
        logging.info(f"{'='*80}\n")
        
        # Create experiment-specific directory
        exp_name = config_path.stem
        exp_dir = self.run_dir / exp_name
        exp_dir.mkdir(parents=True, exist_ok=True)
        
        # Load configs
        model_config, data_config, runtime_config = load_experiment_config(
            str(config_path),
            experiment_dir=str(exp_dir)
        )
        
        # Generate descriptive name
        descriptive_name = generate_experiment_name(model_config, data_config)
        
        # Save config copy to experiment directory
        if rank == 0:
            import shutil
            shutil.copy(str(config_path), str(exp_dir / "config.yaml"))
            
            # Save parsed configs as JSON for reference
            config_info = {
                "experiment_name": exp_name,
                "descriptive_name": descriptive_name,
                "config_file": str(config_path),
                "timestamp": datetime.now().isoformat(),
            }
            with open(exp_dir / "experiment_info.json", 'w') as f:
                json.dump(config_info, f, indent=2)
        
        # Run training
        try:
            losses = test_1f_1b_schedule_vlm_mimo_model_custom_pgs(
                model_config=model_config,
                data_config=data_config,
                runtime_config=runtime_config,
            )
            
            if rank == 0:
                logging.info(f"Rank {rank}: Experiment {exp_name} completed successfully")
                logging.info(f"Rank {rank}: Results saved to: {exp_dir}")
            # dist.destroy_process_group()
                
        except Exception as e:
            logging.error(f"Rank {rank}: Experiment {exp_name} failed with error: {e}")
            if rank == 0:
                # Save error info
                error_info = {
                    "experiment_name": exp_name,
                    "error": str(e),
                    "timestamp": datetime.now().isoformat(),
                }
                with open(exp_dir / "error.json", 'w') as f:
                    json.dump(error_info, f, indent=2)
            # dist.destroy_process_group()
            raise
    
    def aggregate_results(self):
        """Aggregate all experiment CSVs into one combined CSV."""
        rank = dist.get_rank() if dist.is_initialized() else 0
        if rank != 0:
            return
        
        
        import csv
        
        # Find all CSV files in experiment subdirectories
        csv_files = list(self.run_dir.glob("*/metrics_*.csv"))
        
        if not csv_files:
            logging.warning("No CSV files found to aggregate")
            return
        
        # Read all CSVs
        all_rows = []
        fieldnames = None
        
        for csv_file in sorted(csv_files):
            with open(csv_file, 'r') as f:
                reader = csv.DictReader(f)
                if fieldnames is None:
                    fieldnames = reader.fieldnames
                for row in reader:
                    all_rows.append(row)
        
        if not all_rows:
            logging.warning("No data found in CSV files")
            return
        
        # Write combined CSV
        combined_csv = self.run_dir / "all_experiments.csv"
        with open(combined_csv, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_rows)
        
        logging.info(f"Combined results saved to: {combined_csv}")
        logging.info(f"Total experiments: {len(all_rows)}")
    
    def run_all_experiments(self):
        """Run all experiments found in experiments directory."""
        config_files = self.find_config_files()
        
        if not config_files:
            logging.warning(f"No config files found in {self.experiments_dir}")
            return
        
        rank = dist.get_rank() if dist.is_initialized() else 0
        
        if rank == 0:
            logging.info(f"Starting experiment run with {len(config_files)} experiments")
            logging.info(f"Results directory: {self.run_dir}")
        
        for idx, config_path in enumerate(config_files):
            self.run_experiment(config_path, idx, len(config_files))
            
            # Barrier to ensure all ranks complete before moving to next experiment
            if dist.is_initialized():
                dist.barrier()
        
        # Aggregate results
        self.aggregate_results()
        
        if rank == 0:
            logging.info(f"\n{'='*80}")
            logging.info(f"All experiments completed! Results saved to: {self.run_dir}")
            logging.info(f"{'='*80}\n")


def main():
    """Main entry point for experiment runner."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Run ablation experiments from YAML configs")
    parser.add_argument(
        "--experiments-dir",
        type=str,
        default="tests/unit_tests/models/heterogenous_parallel/configs",
        help="Directory containing experiment YAML configs (default: tests/unit_tests/models/heterogenous_parallel/configs)"
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default="logs/mimo_ablations",
        help="Directory to save results (default: logs/mimo_ablations)"
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Run a single config file instead of all configs in experiments-dir"
    )
    
    args = parser.parse_args()
    
    # Initialize distributed training
    Utils.initialize_distributed()
    
    if args.config:
        # Run single experiment
        config_path = Path(args.config)
        if not config_path.exists():
            logging.error(f"Config file not found: {config_path}")
            sys.exit(1)
        
        # Create results directory
        results_dir = Path(args.results_dir) if args.results_dir else config_path.parent / "results"
        results_dir.mkdir(parents=True, exist_ok=True)
        
        runner = ExperimentRunner(
            experiments_dir=str(config_path.parent),
            results_dir=str(results_dir)
        )
        runner.run_experiment(config_path, 0, 1)
    else:
        # Run all experiments in directory
        runner = ExperimentRunner(
            experiments_dir=args.experiments_dir,
            results_dir=args.results_dir
        )
        runner.run_all_experiments()
    
    # Clean up
    dist.destroy_process_group()


if __name__ == "__main__":
    main()

