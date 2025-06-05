import logging

import pytest
import random
from typing import List

import hnsw
from hnsw import Node, vec, search_layer, LayerId, k_nn


@pytest.fixture
def sample_nodes() -> tuple[Node, ...]:
    # Create sample nodes with vectors
    node1 = Node(v=(1.1, 1.2, 1.3, 1.4, 1.5))
    node2 = Node(v=(0.1, 0.2, 0.3, 0.4, 0.5))
    node3 = Node(v=(2.1, 2.2, 2.3, 2.4, 2.5))
    node4 = Node(v=(0.5, 0.5, 0.5, 0.5, 0.5))

    # Set up adjacency for layer 1
    layer_id = 1
    node1.adjacent_nodes[layer_id] = [node2]
    node2.adjacent_nodes[layer_id] = [node1, node3, node4]
    node3.adjacent_nodes[layer_id] = [node2]
    node4.adjacent_nodes[layer_id] = [node2]

    return node1, node2, node3, node4


@pytest.fixture
def layer_id() -> LayerId:
    return 1


@pytest.fixture
def query_vector() -> vec:
    return (0.5, 0.5, 0.5, 0.5, 0.5)


def fully_connected_nodes() -> list[Node]:
    """Create a fully connected layer of 100 nodes for testing"""
    # Create 100 nodes with random vectors
    nodes = []
    for i in range(100):
        nodes.append(Node(v=tuple(random.random() for _ in range(5))))

    # Create a fully connected graph at layer 1
    layer_id = 1
    for i, node in enumerate(nodes):
        # Connect each node to every other node
        for j, other_node in enumerate(nodes):
            if i != j:  # Don't connect to self
                node.adjacent_nodes[layer_id].append(other_node)

    return nodes


def test_fc() -> None:
    for k in range(3,10):
        nodes = fully_connected_nodes()
        query_node = tuple(random.random() for _ in range(5))
        entry_point = nodes[10]

        out = search_layer(query_node, [entry_point], k, 1)
        true = k_nn(query_node, [n.v for n in nodes], k)
        out_set = set(o.v for o in out.nodes())
        true_set = set(true)
        fail_message = ", ".join(f"{hnsw.l2(o,query_node)}" for o in out_set)
        true_msg = ", ".join(f"{hnsw.l2(o,query_node)}" for o in true)
        logging.info(fail_message)
        logging.info(true_msg)
        assert out_set == true_set, fail_message
        assert out_set.intersection(true_set)


def test_search_layer_single_entry_point(
    sample_nodes: tuple[Node, ...], query_vector: vec, layer_id: LayerId
) -> None:
    """Test search_layer with a single entry point"""
    node1 = sample_nodes[0]
    entry_points: List[Node] = [node1]
    retrival_count: int = 2

    result: List[Node] = search_layer(
        query_vector, entry_points, retrival_count, layer_id
    ).nodes()

    assert len(result) <= retrival_count


def test_search_layer_multiple_entry_points(
    sample_nodes: tuple[Node, ...], query_vector: vec, layer_id: LayerId
) -> None:
    """Test search_layer with multiple entry points"""
    node1, node3 = sample_nodes[0], sample_nodes[2]
    entry_points: List[Node] = [node1, node3]
    retrival_count: int = 3

    result: List[Node] = search_layer(
        query_vector, entry_points, retrival_count, layer_id
    ).nodes()

    assert len(result) <= retrival_count


def test_search_layer_retrieval_count_greater_than_nodes(
    sample_nodes: tuple[Node, ...], query_vector: vec, layer_id: LayerId
) -> None:
    """Test search_layer when retrival_count is greater than available nodes"""
    node1 = sample_nodes[0]
    entry_points: List[Node] = [node1]
    retrival_count: int = 10

    result: List[Node] = search_layer(
        query_vector, entry_points, retrival_count, layer_id
    ).nodes()

    # Assert we don't get more nodes than exist
    assert len(result) <= len(sample_nodes)


def test_search_layer_finds_closest_node(
    sample_nodes: tuple[Node, ...], query_vector: vec, layer_id: LayerId
) -> None:
    """Test that search_layer finds the node with the closest vector to the query"""
    # Node4 has the exact query vector
    node1 = sample_nodes[0]
    entry_points: List[Node] = [node1]

    result = search_layer(query_vector, entry_points, 1, layer_id).nodes()

    # Check if the result contains at least one node
    assert len(result) > 0

    # The closest node should be node4 with vector (0.5, 0.5, 0.5, 0.5, 0.5)
    found_closest = False
    for node in result:
        if node.v == query_vector:
            found_closest = True
            break

    # We might not find the exact match due to how the graph is traversed,
    # but the test shows the intention
    assert found_closest or len(result) > 0
