import json
import random
import time
from itertools import product
from typing import List, Dict

from hnsw import HNSW, vec, k_nn


def generate_random_vectors(dim: int = 5, count: int = 1000) -> List[vec]:
    """Generate random vectors with the specified dimension."""
    return [tuple(random.random() for _ in range(dim)) for _ in range(count)]


def evaluate_performance(
    hnsw_instance: HNSW, 
    query_vectors: List[vec], 
    ground_truth: Dict[vec, List[vec]], 
    k: int
) -> Dict[str, float]:
    """
    Evaluate the performance of the HNSW index.
    
    Args:
        hnsw_instance: The HNSW index to evaluate
        query_vectors: List of query vectors
        ground_truth: Dictionary mapping each query vector to its true k nearest neighbors
        k: Number of nearest neighbors to retrieve
        
    Returns:
        Dictionary with performance metrics (recall, query_time)
    """
    total_recall = 0.0
    total_time = 0.0
    
    for query in query_vectors:
        start_time = time.time()
        hnsw_results = hnsw_instance.search(query, k)
        query_time = time.time() - start_time
        
        # Extract vector values from nodes
        hnsw_vectors = {node.v for node in hnsw_results}
        
        # Calculate recall
        true_vectors = set(ground_truth[query])
        recall = len(hnsw_vectors.intersection(true_vectors)) / len(true_vectors)
        
        total_recall += recall
        total_time += query_time
    
    avg_recall = total_recall / len(query_vectors)
    avg_query_time = total_time / len(query_vectors)
    
    return {
        "recall": avg_recall,
        "query_time": avg_query_time
    }


def main():
    # Parameters
    dim = 5  # Dimension of vectors
    dataset_size = 1000  # Number of vectors in the dataset
    query_size = 100  # Number of query vectors
    k = 10  # Number of nearest neighbors to retrieve
    
    # Generate dataset and query vectors
    print(f"Generating {dataset_size} random vectors with dimension {dim}...")
    dataset = generate_random_vectors(dim, dataset_size)
    
    print(f"Generating {query_size} query vectors...")
    query_vectors = generate_random_vectors(dim, query_size)
    
    # Compute ground truth (exact k-NN) for each query vector
    print("Computing ground truth nearest neighbors...")
    ground_truth = {}
    for query in query_vectors:
        ground_truth[query] = k_nn(query, dataset, k)
    
    # Grid search parameters
    param_grid = {
        "max_connections": [10, 20, 50],
        "construction_search_max": [10, 20, 40],
    }
    
    # Generate all parameter combinations
    param_combinations = list(product(
        param_grid["max_connections"],
        param_grid["construction_search_max"],
    ))
    
    print(f"Running grid search with {len(param_combinations)} parameter combinations...")
    
    # Store results
    results = []
    
    # Run grid search
    for max_conn, constr_search in param_combinations:
        print(f"\nTesting parameters: max_conn={max_conn}, constr_search={constr_search}")
        
        # Create and build HNSW index
        hnsw_instance = HNSW(
            max_connections=max_conn,
            construction_search_max=constr_search,
        )
        
        # Build index
        build_start_time = time.time()
        for vector in dataset:
            hnsw_instance.insert(vector)
        build_time = time.time() - build_start_time
        
        print(f"  Build time: {build_time:.4f} seconds")
        
        # Evaluate performance
        performance = evaluate_performance(hnsw_instance, query_vectors, ground_truth, k)
        
        # Store results
        result = {
            "max_connections": max_conn,
            "construction_search_max": constr_search,
            "build_time": build_time,
            "recall": performance["recall"],
            "query_time": performance["query_time"]
        }
        
        results.append(result)
        
        print(f"  Recall: {performance['recall']:.4f}")
        print(f"  Avg query time: {performance['query_time'] * 1000:.4f} ms")
    
    with open("results.json", "w") as f:
        json.dump(results, f)
    # Find best parameters
    best_recall_result = max(results, key=lambda x: x["recall"])
    best_speed_result = min(results, key=lambda x: x["query_time"])
    
    print("\n=== Grid Search Results ===")
    print("\nBest parameters for recall:")
    print(f"  max_connections: {best_recall_result['max_connections']}")
    print(f"  construction_search_max: {best_recall_result['construction_search_max']}")
    print(f"  Recall: {best_recall_result['recall']:.4f}")
    print(f"  Build time: {best_recall_result['build_time']:.4f} seconds")
    print(f"  Avg query time: {best_recall_result['query_time'] * 1000:.4f} ms")
    
    print("\nBest parameters for query speed:")
    print(f"  max_connections: {best_speed_result['max_connections']}")
    print(f"  construction_search_max: {best_speed_result['construction_search_max']}")
    print(f"  Recall: {best_speed_result['recall']:.4f}")
    print(f"  Build time: {best_speed_result['build_time']:.4f} seconds")
    print(f"  Avg query time: {best_speed_result['query_time'] * 1000:.4f} ms")
