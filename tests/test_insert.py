import random
import pytest
import heapq

from hnsw import HNSW, vec, l2
import hnsw


@pytest.fixture
def insert_points() -> list[vec]:
    out = []
    for _ in range(1000):
        out.append(tuple(random.random() for _ in range(5)))
    return out


def build_tree(insert_points) -> HNSW:
    tree = hnsw.HNSW(
        50,
        20,
    )
    for point in insert_points:
        tree.insert(point)
    return tree


def test_insert(insert_points) -> None:
    tree = hnsw.HNSW(
        10,
        10,
    )
    for point in insert_points:
        tree.insert(point)


def test_hnsw_vs_knn(insert_points) -> None:
    """Test HNSW search against brute force k-NN implementation"""
    # Create query vectors (different from insert_points)
    built_tree = build_tree(insert_points)
    query_vectors = [tuple(random.random() for _ in range(5)) for _ in range(5)]

    # Number of neighbors to retrieve
    k = 3

    for query in query_vectors:
        # Get approximate nearest neighbors using HNSW
        hnsw_results = built_tree.search(query, k)
        hnsw_vectors = {node.v for node in hnsw_results}

        # Get exact nearest neighbors using k_nn function
        exact_nn = set(hnsw.k_nn(query, insert_points, k))

        # Calculate the recall (percentage of true nearest neighbors found by HNSW)
        common_vectors = hnsw_vectors.intersection(exact_nn)
        recall = len(common_vectors) / len(exact_nn)

        assert recall >= 0.6
        assert len(hnsw_results) == k
