from __future__ import annotations

from collections import defaultdict
import heapq
import logging
import math
import random
from dataclasses import dataclass, field
from typing import Iterable, Self

log = logging.getLogger(__name__)
type vec = tuple[float, float, float, float, float]


def l2(a: vec, b: vec) -> float:
    dist = sum((a_ - b_) ** 2 for a_, b_ in zip(a, b))
    return dist


type LayerId = int


@dataclass
class Node:
    v: vec
    adjacent_nodes: dict[LayerId, list[Node]] = field(
        default_factory=lambda: defaultdict(list)
    )

    def __hash__(self) -> int:
        return hash(self.v)

    def layer_neigbours(self, id: LayerId) -> list[Node]:
        return self.adjacent_nodes[id]


class MinDistanceHeap:
    def __init__(self, query_node: vec) -> None:
        self._nodes: list[PrioritizedNode] = []
        self.query_node = query_node

    def heapify(self, d: Iterable[Node]) -> Self:
        for i in d:
            self.push(i)
        return self

    def push(self, n: Node) -> None:
        heapq.heappush(self._nodes, PrioritizedNode.from_node(n, self.query_node))

    def pop_closest(self) -> tuple[Node, float]:
        n = heapq.heappop(self._nodes)
        return n.n, n.distance

    def get_closest(self) -> Node:
        return self._nodes[0].n

    def get_closest_dist(self) -> float:
        return self._nodes[0].distance

    def __len__(self) -> int:
        return len(self._nodes)

    def nodes(self) -> list[Node]:
        return [n.n for n in self._nodes]


class MaxDistanceHeap:
    def __init__(self, query_node: vec) -> None:
        self._nodes: list[PrioritizedNode] = []
        self.query_node = query_node

    def push(self, n: Node) -> None:
        heapq.heappush(
            self._nodes, PrioritizedNode.from_node_inverse(n, self.query_node)
        )

    def heapify(self, d: Iterable[Node]) -> Self:
        for i in d:
            self.push(i)
        return self

    def pop_furthest(self) -> Node:
        return heapq.heappop(self._nodes).n

    def get_furthest(self) -> Node:
        return self._nodes[0].n

    def get_furthest_dist(self) -> float:
        return -self._nodes[0].distance

    def __len__(self) -> int:
        return len(self._nodes)

    def nodes(self) -> list[Node]:
        return [n.n for n in self._nodes]


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


def select_neighbours_simple(
    query_node: vec,
    candidate_nodes: list[Node],
    retrival_count: int,
    layer_id: LayerId,
) -> list[Node]:
    max_heap = MaxDistanceHeap(query_node)

    for n in candidate_nodes:
        max_heap.push(n)

        if len(max_heap) > retrival_count:
            max_heap.pop_furthest()
    return max_heap.nodes()


def select_neighbours_heuristic(
    query_node: vec,
    candidate_nodes: list[Node],
    retrival_count: int,
    layer_id: LayerId,
    extend_candidates: bool = False,
    keep_pruned_connections: bool = False,
) -> list[Node]:
    candidate_vecs = {n.v for n in candidate_nodes}
    candidate_queue = MinDistanceHeap(query_node).heapify(candidate_nodes)
    #
    out_queue = MaxDistanceHeap(query_node)
    #
    if extend_candidates:
        #
        for candidate_node in candidate_nodes:
            #
            for c_neigbour in candidate_node.layer_neigbours(layer_id):
                #
                if c_neigbour.v in candidate_vecs:
                    continue
                #
                candidate_queue.push(c_neigbour)
                candidate_vecs.add(c_neigbour.v)

    discard_queue = MinDistanceHeap(query_node)

    while candidate_queue and len(out_queue) <= retrival_count:
        new_candidate, new_candidate_distance = candidate_queue.pop_closest()

        if not out_queue:
            out_queue.push(new_candidate)

        elif new_candidate_distance < out_queue.get_furthest_dist():
            out_queue.push(new_candidate)

        else:
            discard_queue.push(new_candidate)

    if keep_pruned_connections:
        while len(discard_queue) and len(out_queue) < retrival_count:
            out_queue.push(discard_queue.get_closest())
    return out_queue.nodes()


def search_layer(
    query_node: vec, entry_points: list[Node], retrival_count: int, layer_id: LayerId
) -> MaxDistanceHeap:
    visited_elements = set(entry_points)

    candidate_nodes = MinDistanceHeap(query_node).heapify(entry_points)
    found_nodes = MaxDistanceHeap(query_node).heapify(entry_points)
    while candidate_nodes:
        #
        closest_node, closest_node_dist = candidate_nodes.pop_closest()
        if closest_node_dist > found_nodes.get_furthest_dist():
            break

        for neighbour in closest_node.layer_neigbours(layer_id):
            #
            if neighbour in visited_elements:
                continue
            #
            #
            visited_elements.add(neighbour)
            neighbour_distance = l2(neighbour.v, query_node)
            #
            if (
                neighbour_distance < found_nodes.get_furthest_dist()
                or len(found_nodes) < retrival_count
            ):
                #
                candidate_nodes.push(closest_node)
                found_nodes.push(closest_node)
                #
                while len(found_nodes) > retrival_count:
                    found_nodes.pop_furthest()

    while len(found_nodes) > retrival_count:
        found_nodes.pop_furthest()

    return found_nodes


class HNSW:
    def __init__(
        self,
        max_connections: int,
        construction_search_max: int,
        normalisation_factor: float = 0.5,
        number_of_connections: int = 5,
    ) -> None:
        pass
        self.max_connections: int = max_connections
        self.construction_search_max: int = construction_search_max
        self.entry_point = None
        self.top_layer_id = 0
        self.normalisation_factor: float = normalisation_factor
        self.number_of_connections: int = number_of_connections

    def sample_layer_id(self) -> LayerId:
        x = -math.log(random.random()) * self.normalisation_factor
        return math.floor(x)

    def insert(self, query_node: vec) -> None:
        target_layer = self.sample_layer_id()
        # Handle no nodes.
        if self.entry_point is None:
            self.top_layer_id = target_layer
            self.entry_point = Node(query_node)
            return
        top_layer_id = self.top_layer_id
        log.debug(f"Insertion target layer:{target_layer}, top layer:{top_layer_id}")
        step = 1 if top_layer_id > target_layer else -1
        new_node = Node(query_node)
        neighbours = []
        entry_point = self.entry_point
        # Climb up or down
        # Greedily searching the layer for the closest node.
        for layer_id in range(top_layer_id, target_layer + 1, step):
            [entry_point] = search_layer(query_node, [entry_point], 1, layer_id).nodes()
            log.debug("Found entry_point")

        entry_points = [entry_point]
        # Descend down the tree
        for layer_id in range(min(top_layer_id, target_layer), 0, -1):
            candiates = search_layer(
                query_node,
                entry_points,
                self.construction_search_max,
                layer_id,
            ).nodes()
            if not candiates:
                continue
            neighbours = select_neighbours_simple(
                query_node, candiates, self.number_of_connections, layer_id=layer_id
            )
            # Form links at this layer
            for neighbour in neighbours:
                new_node.layer_neigbours(layer_id).append(neighbour)
                neighbour.layer_neigbours(layer_id).append(new_node)
                if (
                    len(ns := neighbour.layer_neigbours(layer_id))
                    > self.max_connections
                ):
                    pruned_neigbourhood = select_neighbours_simple(
                        neighbour.v, ns, self.max_connections, layer_id
                    )
                    neighbour.adjacent_nodes[layer_id] = pruned_neigbourhood
            entry_points = candiates

        if target_layer > top_layer_id:
            entry_point = new_node
            self.top_layer_id = target_layer

    def search(self, query_node: vec, k: int, max_candiates: int = 20) -> list[Node]:
        if self.entry_point is None:
            raise ValueError("No nodes in the tree.")
        entry_point = self.entry_point
        for layer_id in range(self.top_layer_id, 1, -1):
            log.debug("traversing to layer %s", layer_id)
            candidates = search_layer(query_node, [entry_point], 1, layer_id)
            entry_point = max(candidates._nodes).n
        candidates = search_layer(query_node, [entry_point], max_candiates, 0)
        while len(candidates) > k:
            candidates.pop_furthest()
        return candidates.nodes()


def k_nn(a: vec, b: list[vec], k: int) -> list[vec]:
    """
    Find the k nearest neighbors of vector a in list b.

    Args:
        a: Query vector
        b: List of vectors to search
        k: Number of nearest neighbors to return

    Returns:
        List of k vectors from b that are closest to a
    """
    # Calculate distances between a and each vector in b
    distances = [(l2(a, vector), vector) for vector in b]

    # Use heapq to efficiently find k smallest distances
    nearest = heapq.nsmallest(k, distances, key=lambda x: x[0])

    # Return just the vectors (not the distances)
    return [vector for _, vector in nearest]
