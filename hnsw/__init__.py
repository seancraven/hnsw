from __future__ import annotations

import heapq
from copy import copy
from dataclasses import dataclass, field
from typing import Self

type vec = tuple[float, float, float, float, float]


def l2(a: vec, b: vec) -> float:
    dist = sum((a_ - b_) ** 2 for a_, b_ in zip(a, b))
    return dist


@dataclass
class Node:
    v: vec
    layer_id: int
    adjacent_nodes: list[Node]

    def __hash__(self) -> int:
        return hash((self.layer_id, *self.v))


@dataclass(order=True)
class PrioritizedNode:
    distance: float
    n: Node = field(compare=False)

    @classmethod
    def from_node(cls, n: Node, query_node: vec) -> Self:
        return cls(distance=l2(n.v, query_node), n=n)

    @classmethod
    def from_node_inverse(cls, n: Node, query_node: vec) -> Self:
        return cls(distance=-l2(n.v, query_node), n=n)


@dataclass
class NodeLayer:
    nodes: list[Node]
    layer_id: int

    def search_layer(
        self, query_node: vec, entry_points: list[Node], retrival_count: int
    ) -> list[Node]:
        for node in entry_points:
            assert self.layer_id == node.layer_id
        visited_elements = set(entry_points)

        # Make queues.
        candidate_nodes = [
            PrioritizedNode.from_node(e, query_node) for e in entry_points
        ]
        heapq.heapify(candidate_nodes)
        found_nodes = [
            PrioritizedNode.from_node_inverse(e, query_node) for e in entry_points
        ]
        heapq.heapify(found_nodes)
        # Perform search
        while candidate_nodes:
            closest_node = heapq.heappop(candidate_nodes)
            furthest_found_node_distance = -found_nodes[0].distance
            if closest_node.distance > furthest_found_node_distance:
                break
            for neighbour in closest_node.n.adjacent_nodes:
                if neighbour in visited_elements:
                    continue
                visited_elements.add(neighbour)
                furthest_found_node_distance = -found_nodes[0].distance
                neighbour_distance = l2(neighbour.v, query_node)
                if (
                    neighbour_distance < furthest_found_node_distance
                    or len(found_nodes) < retrival_count
                ):
                    heapq.heappush(
                        candidate_nodes,
                        PrioritizedNode.from_node(neighbour, query_node),
                    )
                    heapq.heappush(
                        found_nodes,
                        PrioritizedNode.from_node_inverse(neighbour, query_node),
                    )
                    # Drop if node count is higher.
                    while len(found_nodes) > retrival_count:
                        heapq.heappop(found_nodes)

        while len(found_nodes) > retrival_count:
            heapq.heappop(found_nodes)

        return [cn.n for cn in found_nodes]


@dataclass
class HNSW:
    layers: list[NodeLayer]
