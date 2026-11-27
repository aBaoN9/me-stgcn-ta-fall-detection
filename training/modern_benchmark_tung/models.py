"""Explicit registry, with independent adapter modules and per-model locks."""
from importlib import import_module
from pathlib import Path

from training.backbone_benchmark.models import A3, build_model as build_a3

ADAPTERS = {
    "cascadeformer_v2_ce": "cascade",
    "dha_egcn_v2_ce": "dha",
    "hyper_gcn_v2_ce": "hyper",
    "proto_gcn_v2_ce": "proto",
    "infogcnpp_v2_ce": "infogcn",
}
MODEL_IDS = (A3, *ADAPTERS)


def adapter_source(model):
    return Path(__file__).with_name("adapters") / f"{ADAPTERS[model]}.py"


def build_model(model, dropout=0.3):
    if model == A3:
        return build_a3(A3, dropout)
    if model not in ADAPTERS:
        raise ValueError(f"Unknown model ID: {model}")
    return import_module(f".adapters.{ADAPTERS[model]}", __package__).Model(dropout)
