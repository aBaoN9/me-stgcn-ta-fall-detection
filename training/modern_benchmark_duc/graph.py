"""MediaPipe 33-joint graph shared by every adapted backbone.

Edges are imported from the A3 baseline (training/component_ablation/models.py) so that every model sees
exactly the skeleton A3 sees. No NTU/COCO adjacency is used anywhere. MediaPipe's pose graph has three
components (face 0-8, mouth 9-10, body 11-32); helpers below make that explicit instead of hiding it.
"""
from collections import deque

import numpy as np

from training.component_ablation.models import EDGES

NUM_JOINTS = 33
# Hop distance is measured from these roots, one root set per connected component.
ROOTS = (11, 12, 23, 24, 0, 9, 10)


def neighbours():
    table = {joint: set() for joint in range(NUM_JOINTS)}
    for left, right in EDGES:
        table[left].add(right)
        table[right].add(left)
    return table


def shortest_paths():
    """All-pairs hop distance on the A3 edge list; unreachable pairs are np.inf."""
    table = neighbours()
    distance = np.full((NUM_JOINTS, NUM_JOINTS), np.inf)
    for source in range(NUM_JOINTS):
        distance[source, source] = 0
        queue = deque([source])
        while queue:
            node = queue.popleft()
            for other in table[node]:
                if distance[source, other] == np.inf:
                    distance[source, other] = distance[source, node] + 1
                    queue.append(other)
    return distance


def root_depth():
    distance = shortest_paths()
    depth = distance[list(ROOTS)].min(axis=0)
    if not np.isfinite(depth).all():
        raise ValueError("Every joint must reach a root")
    return depth


def directed_edges():
    """(child, parent) pairs: parent is closer to the component root. Equal-depth edges go both ways."""
    depth = root_depth()
    inward = []
    for left, right in EDGES:
        if depth[left] > depth[right]:
            inward.append((left, right))
        elif depth[left] < depth[right]:
            inward.append((right, left))
        else:
            inward += [(left, right), (right, left)]
    return inward


def edge2mat(link, num_node=NUM_JOINTS):
    matrix = np.zeros((num_node, num_node))
    for source, target in link:
        matrix[target, source] = 1
    return matrix


def normalize_digraph(matrix):
    degree = matrix.sum(0)
    inverse = np.zeros_like(degree)
    inverse[degree > 0] = degree[degree > 0] ** -1
    return matrix @ np.diag(inverse)


def spatial_graph():
    """ST-GCN 'spatial' labelling (identity, inward, outward) as used by CTR-GCN-family code: (3,33,33)."""
    inward = directed_edges()
    outward = [(parent, child) for child, parent in inward]
    identity = edge2mat([(joint, joint) for joint in range(NUM_JOINTS)])
    return np.stack([identity, normalize_digraph(edge2mat(inward)), normalize_digraph(edge2mat(outward))]).astype(np.float32)


def hop_matrix():
    """Hop distance with a dedicated bucket (max_hop + 1) for joints in different components."""
    distance = shortest_paths()
    finite = distance[np.isfinite(distance)]
    hops = np.where(np.isfinite(distance), distance, finite.max() + 1)
    return hops.astype(np.int64)


# SkateFormer groups joints into equal-size parts. Six anatomical parts x six slots; -1 marks padding.
SKATE_PARTS = (
    (12, 14, 16, 18, 20, 22),  # right arm with shoulder and hand points
    (11, 13, 15, 17, 19, 21),  # left arm
    (24, 26, 28, 30, 32, -1),  # right leg with hip and foot points
    (23, 25, 27, 29, 31, -1),  # left leg
    (0, 4, 5, 6, 8, 10),       # nose, right eye, right ear, right mouth corner
    (1, 2, 3, 7, 9, -1),       # left eye, left ear, left mouth corner
)


def skate_layout():
    order = [joint for part in SKATE_PARTS for joint in part]
    real = [joint for joint in order if joint >= 0]
    if sorted(real) != list(range(NUM_JOINTS)) or len(set(len(part) for part in SKATE_PARTS)) != 1:
        raise ValueError("SkateFormer layout must cover 33 joints in equal-size parts")
    return np.asarray(order, dtype=np.int64)
