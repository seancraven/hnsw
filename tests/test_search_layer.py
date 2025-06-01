import pytest
from typing import List
from hnsw import NodeLayer, Node, vec


@pytest.fixture
def sample_nodes() -> tuple[Node, ...]:
    # These aren't entirely correct.
    # The node locations are determined by the insertion algorithm which means that the graph is
    # well defined.
    node1 = Node(v=(1.1, 1.2, 1.3, 1.4, 1.5), layer_id=1, adjacent_nodes=[])
    node2 = Node(v=(0.1, 0.2, 0.3, 0.4, 0.5), layer_id=1, adjacent_nodes=[])
    node3 = Node(v=(2.1, 2.2, 2.3, 2.4, 2.5), layer_id=1, adjacent_nodes=[])
    node4 = Node(v=(0.5, 0.5, 0.5, 0.5, 0.5), layer_id=1, adjacent_nodes=[])

    # Set up adjacency
    node1.adjacent_nodes = [node2]
    node2.adjacent_nodes = [node1, node3, node4]
    node3.adjacent_nodes = [node2]
    node4.adjacent_nodes = [node2]

    return node1, node2, node3, node4


@pytest.fixture
def node_layer(sample_nodes: tuple[Node, ...]) -> NodeLayer:
    return NodeLayer(nodes=list(sample_nodes), layer_id=1)


@pytest.fixture
def query_vector() -> vec:
    return (0.5, 0.5, 0.5, 0.5, 0.5)


def test_search_layer_single_entry_point(
    node_layer: NodeLayer, sample_nodes: tuple[Node, ...], query_vector: vec
) -> None:
    """Test search_layer with a single entry point"""
    node1 = sample_nodes[0]
    entry_points: List[Node] = [node1]
    retrival_count: int = 2

    result: List[Node] = node_layer.search_layer(
        query_vector, entry_points, retrival_count
    )

    assert len(result) == retrival_count


def test_search_layer_multiple_entry_points(
    node_layer: NodeLayer, sample_nodes: tuple[Node, ...], query_vector: vec
) -> None:
    """Test search_layer with multiple entry points"""
    node1, node3 = sample_nodes[0], sample_nodes[2]
    entry_points: List[Node] = [node1, node3]
    retrival_count: int = 3

    result: List[Node] = node_layer.search_layer(
        query_vector, entry_points, retrival_count
    )

    assert len(result) == retrival_count


def test_search_layer_retrieval_count_greater_than_nodes(
    node_layer: NodeLayer, sample_nodes: tuple[Node, ...], query_vector: vec
) -> None:
    """Test search_layer when retrival_count is greater than available nodes"""
    node1 = sample_nodes[0]
    entry_points: List[Node] = [node1]
    retrival_count: int = 10

    result: List[Node] = node_layer.search_layer(
        query_vector, entry_points, retrival_count
    )

    # Assert we don't get more nodes than exist
    assert len(result) == len(node_layer.nodes)


def test_retrieval(
    node_layer: NodeLayer, sample_nodes: tuple[Node, ...], query_vector: vec
) -> None:
    for n in sample_nodes:
        result: List[Node] = node_layer.search_layer(query_vector, [n], 1)
        assert result[0].v == query_vector
