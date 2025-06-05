# HNSW: Hierarchical Navigable Small World

RAG good, and FAISS fast. I make slow bad version in python.
Maybe make fast Rust version I don't know.
Not sure if complexity is actually as it should be, there might be some silly implmentation details to iron out.

## Algorithm Overview

HNSW works by creating a hierarchical graph structure where:

- Each node represents a vector in the dataset
- Each layer in the hierarchy contains a subset of nodes from lower layers
- Higher layers form a coarser graph, allowing for efficient navigation
- The search process starts at the top layer and navigates down

## Usage Example

```python
from hnsw import HNSW

# Create an HNSW index
index = HNSW(
    max_connections=20,       # Maximum connections per node
    construction_search_max=40,  # Maximum candidates during construction
)

# Add vectors to the index
vectors = [(0.1, 0.2, 0.3, 0.4, 0.5), (0.2, 0.3, 0.4, 0.5, 0.6)]
for vec in vectors:
    index.insert(vec)

# Search for nearest neighbors
query = (0.15, 0.25, 0.35, 0.45, 0.55)
results = index.search(query, k=5)  # Find 5 nearest neighbors
```

## Performance Evaluation

The repository includes a performance evaluation framework that:

1. Generates random vectors for testing
2. Computes exact k-nearest neighbors as ground truth
3. Performs grid search over HNSW parameters
4. Evaluates performance metrics (recall, query time, build time)
5. Identifies optimal parameter configurations

Run the evaluation:

```bash
uv run evaluate
```

This will generate a `results.json` file with detailed performance metrics for different parameter combinations.

## Key Parameters

- `max_connections`: Maximum number of connections per node
- `construction_search_max`: Maximum number of candidates to consider during index construction
- `normalisation_factor`: Controls the distribution of nodes across layers (default: 0.5)
- `number_of_connections`: Initial number of connections to create for each new node (default: 5)

## References

- Original HNSW paper: Malkov, Y. A., & Yashunin, D. A. (2018). Efficient and robust approximate nearest neighbor search using hierarchical navigable small world graphs. IEEE transactions on pattern analysis and machine intelligence, 42(4), 824-836.

