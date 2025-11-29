# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

import warnings
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from megatron.core.transformer.spec_utils import ModuleSpec


@dataclass
class ColocatedCommConfig:
    """Configuration for colocated module communication.
    
    Used when encoder and LLM are colocated (same GPUs) but have different
    TP/DP configurations with PP=1.
    
    Args:
        module_to_grid_map: Mapping of module names to their HyperCommGrids.
            Example: {'images': encoder_grid, 'language_module': llm_grid}
        topology: DAG defining data flow between modules.
            Example: {'images': ['language_module'], 'language_module': []}
        dim_mapping: Mapping of logical dimensions to tensor axes.
            Defaults to {'b': 0, 's': 1, 'h': 2} for [batch, seq, hidden].
    """
    module_to_grid_map: Dict = field(default_factory=dict)
    topology: Dict[str, List[str]] = field(default_factory=dict)
    dim_mapping: Dict[str, int] = field(default_factory=lambda: {'b': 0, 's': 1, 'h': 2})


@dataclass
class MimoModelConfig:
    """Configuration for a multi-modal model.

    Args:
        language_model_spec (ModuleSpec):
            Specification for the language model
        modality_submodules_spec (Dict[str, ModuleSpec]):
            Dictionary mapping modality names to their submodule specifications
        special_token_ids (Dict[str, int]):
            Dictionary mapping modality names to their special token IDs.
            For example, {"vision": -200, "audio":32000}, these represent placeholders
            in the input_ids to insert the modality embeddings at the correct positions.
    """

    warnings.warn(
        "MimoModelConfig is experimental and still under active development. "
        "The API may change without notice in future releases.",
        category=UserWarning,
        stacklevel=2,
    )

    language_model_spec: ModuleSpec = field(default_factory=ModuleSpec)
    modality_submodules_spec: Dict[str, ModuleSpec] = field(default_factory=dict)
    special_token_ids: Dict[str, int] = field(default_factory=dict)
    colocated_comm_config: Optional[ColocatedCommConfig] = None
