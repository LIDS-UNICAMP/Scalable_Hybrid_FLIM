"""Inicialização abstrata de kernels (SPiFiL-0..4) — módulo opcional, fora da produção.

Interface: ``ga_search.abstract_init.base`` (KernelInitializer, TrainData, InitContext, validate_init,
assemble_block, make_provenance, registro). Adaptadores SPiFiL-0 (S0, S0+, S0-data): ``spifil0``.
Runner de ablação: ``python -m ga_search.abstract_init.run_ablation``.
"""
from ga_search.abstract_init.base import (  # noqa: F401
    INTERFACE_VERSION, InfeasibleCandidate, InitBudget, InitContext, KernelInitializer, ParentInfo,
    TrainData, assemble_block, available, conv_param_keys, get_initializer, make_provenance, register,
    split_state, validate_init,
)
