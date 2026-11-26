"""Model registry for Duc's five assigned IDs plus the A3 control (GB-20261124-001)."""
from training.component_ablation.models import SkeletonModel

A3 = "A3_motion_attention"
SKATEFORMER = "skateformer_v2_ce"
BLOCKGCN = "block_gcn_v2_ce"
DEGCN = "de_gcn_v2_ce"
JT_GRAPHFORMER = "jt_graphformer_v2_ce"
DS_GCN = "ds_gcn_v2_ce"
ASSIGNED = (SKATEFORMER, BLOCKGCN, DEGCN, JT_GRAPHFORMER, DS_GCN)
MODEL_IDS = (A3,) + ASSIGNED

# Upstream provenance written into every lock. DS-GCN has no licence file: its code is not copied or ported.
UPSTREAM = {
    SKATEFORMER: dict(repo="https://github.com/KAIST-VICLab/SkateFormer", commit="0249e19d96b33e9150760129bbede56464126ae7",
                      license="MIT", files=["model/SkateFormer.py", "config/train/nw_ucla/SkateFormer_j.yaml"]),
    BLOCKGCN: dict(repo="https://github.com/ZhouYuxuanYX/BlockGCN", commit="7b45971a931c5b0ba60f3a7ba11a91e544de54fb",
                   license="Apache-2.0", files=["model/BlockGCN.py", "graph/tools.py"],
                   also="torch_topological StructureElementLayer, aidos-lab/pytorch-topological 8aca6dbb87d0b03902df657528f53c06184a63c6, BSD-3-Clause"),
    DEGCN: dict(repo="https://github.com/WoominM/DeGCN_pytorch", commit="a0a261fbf4369095083a7b22fa371fb9448f69a8",
                license="MIT", files=["model/degcn.py", "model/modules.py", "config/nturgbd-cross-subject/default.yaml"],
                note="DeSGC is not in the public release; released Basic_Block (CTR_GC + DeTGC) is ported"),
    JT_GRAPHFORMER: dict(repo="https://github.com/ZhengYaolin/JT-GraphFormer", commit="71fb81d177c151f029403cff9b7f2b0d93f8e01c",
                         license="MIT", files=["model/gf2_new.py", "model/pos_embed.py", "config/gf2.yaml", "config/nucla.yaml"]),
    DS_GCN: dict(repo="https://github.com/davelailai/DS-GCN", commit="ffc42f0f5ddd5fd7e06cb2d7bd7e6c5a34fec6ed",
                 license="NONE FOUND (no LICENSE file; README says 'generated based on pyskl')"),
}

# Architecture hyperparameters locked before outer-test; author configs except where the contract forces a change.
ARCHITECTURE = {
    A3: dict(channels=8, attention=True, dropout=0.3),
    SKATEFORMER: dict(in_channels=8, num_classes=2, depths=(2, 2, 2, 2), channels=(96, 192, 192, 192), embed_dim=96,
                      num_frames=32, kernel_size=7, num_heads=32, partition=((4, 6), (4, 6), (4, 6), (4, 6)),
                      attn_drop=0.3, head_drop=0.3, drop=0.3, rel=True, mlp_ratio=1.0),
    BLOCKGCN: dict(in_channels=8, num_classes=2, num_point=33, num_person=1, drop_out=0.3, topology_channels=(0, 1)),
    DEGCN: dict(in_channels=8, num_classes=2, num_point=33, num_person=1, eta=4, num_stream=2, drop_out=0.3, base_frame=32),
    JT_GRAPHFORMER: dict(num_channels=8, num_classes=2, num_joints=33, num_frames=32, len_parts=8, num_heads=3,
                         kernel_size=(3, 5), use_pes=True, att_drop=0.3, dropout=0.3, dropout2d=0.3),
}


class BlockedModel(RuntimeError):
    pass


def build_model(model_id):
    if model_id == A3:
        return SkeletonModel(**ARCHITECTURE[A3])
    if model_id == SKATEFORMER:
        from training.modern_benchmark_duc.skateformer import SkateFormerV2
        return SkateFormerV2(**ARCHITECTURE[SKATEFORMER])
    if model_id == BLOCKGCN:
        from training.modern_benchmark_duc.blockgcn import BlockGCNV2
        return BlockGCNV2(**ARCHITECTURE[BLOCKGCN])
    if model_id == DEGCN:
        from training.modern_benchmark_duc.degcn import DeGCNV2
        return DeGCNV2(**ARCHITECTURE[DEGCN])
    if model_id == JT_GRAPHFORMER:
        from training.modern_benchmark_duc.jt_graphformer import JTGraphFormerV2
        return JTGraphFormerV2(**ARCHITECTURE[JT_GRAPHFORMER])
    if model_id == DS_GCN:
        raise BlockedModel("ds_gcn_v2_ce is BLOCKED: davelailai/DS-GCN@ffc42f0 has no licence file, so its code "
                           "cannot be copied or ported (GB-20261124-001 section 6). No substitute model is trained.")
    raise ValueError(f"Unknown model id {model_id!r}; allowed: {', '.join(MODEL_IDS)}")


def logits_of(model, values):
    output = model(values)
    return output[0] if isinstance(output, tuple) else output
