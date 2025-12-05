## Running Experiments

```bash
# Run single experiment (e.g., 3B LLM baseline)
uv run python -m torch.distributed.run --nproc_per_node=8 \
    tests/unit_tests/models/heterogenous_parallel/experiment_runner.py \
    --config tests/unit_tests/models/heterogenous_parallel/configs/ablations/llm_3b/baseline.yaml

# Run all experiments in a folder (e.g., all 3B LLM configs)
uv run python -m torch.distributed.run --nproc_per_node=8 \
    tests/unit_tests/models/heterogenous_parallel/experiment_runner.py \
    --experiments-dir tests/unit_tests/models/heterogenous_parallel/configs/ablations/llm_3b/
```
