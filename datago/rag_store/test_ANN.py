"""
test_ANN.py - Test script for approximate nearest neighbor search

Generates dummy position embeddings and tests the ANN search functionality.
"""

import numpy as np
from ANN import PositionEmbeddingStore


def generate_dummy_embedding(idx: int, base_winrate: float = 0.5,
                             base_score: float = 0.0) -> dict:
    """
    Generate a dummy position embedding matching GoStateEmbedding.to_dict() format.

    Args:
        idx: Index for unique identification
        base_winrate: Base winrate value (will add small random noise)
        base_score: Base score lead value (will add small random noise)

    Returns:
        Dict matching GoStateEmbedding.to_dict() format
    """
    # Add small random variations
    winrate = np.clip(base_winrate + np.random.normal(0, 0.05), 0.0, 1.0)
    score_lead = base_score + np.random.normal(0, 2.0)

    return {
        'sym_hash': f'sym_hash_{idx}',
        'state_hash': f'state_hash_{idx}',
        'policy': np.random.rand(362).tolist(),  # 19x19 + pass
        'ownership': np.random.rand(361).tolist(),  # 19x19
        'winrate': float(winrate),
        'score_lead': float(score_lead),
        'move_infos': [
            {
                'move': 'Q4',
                'visits': np.random.randint(100, 1000),
                'winrate': np.random.rand(),
                'prior': np.random.rand(),
            }
        ],
        'komi': 7.5,
        'query_id': f'query_{idx}',
    }


def generate_similar_embedding(template: dict, idx: int,
                               similarity: float = 0.9) -> dict:
    """
    Generate an embedding similar to a template.

    Args:
        template: Template embedding to base on
        idx: Index for unique identification
        similarity: How similar to make it (0.0 to 1.0, where 1.0 is identical)

    Returns:
        Dict matching GoStateEmbedding.to_dict() format
    """
    # Mix template with random noise based on similarity
    policy = np.array(template['policy'])
    ownership = np.array(template['ownership'])

    # Add noise inversely proportional to similarity
    noise_factor = 1.0 - similarity
    policy = (similarity * policy + noise_factor * np.random.rand(362))
    ownership = (similarity * ownership + noise_factor * np.random.rand(361))

    # Normalize to valid probability distributions
    policy = policy / policy.sum()

    winrate = template['winrate'] + np.random.normal(0, 0.02 * noise_factor)
    winrate = np.clip(winrate, 0.0, 1.0)

    score_lead = template['score_lead'] + np.random.normal(0, 1.0 * noise_factor)

    return {
        'sym_hash': f'similar_hash_{idx}',
        'state_hash': f'similar_state_{idx}',
        'policy': policy.tolist(),
        'ownership': ownership.tolist(),
        'winrate': float(winrate),
        'score_lead': float(score_lead),
        'move_infos': template.get('move_infos', []),
        'komi': template['komi'],
        'query_id': f'similar_query_{idx}',
    }


def main():
    """Test the ANN search with dummy data."""

    print("=" * 70)
    print("Testing Approximate Nearest Neighbor Search with HNSW")
    print("=" * 70)

    # Generate a collection of dummy embeddings
    print("\n1. Generating dummy embeddings...")
    num_embeddings = 1000  # Increased to better test approximate search
    embeddings = []

    # Create diverse embeddings
    for i in range(num_embeddings):
        # Vary winrate and score across the dataset
        winrate = 0.3 + (i / num_embeddings) * 0.4  # Range from 0.3 to 0.7
        score = -5.0 + (i / num_embeddings) * 10.0  # Range from -5 to 5
        embeddings.append(generate_dummy_embedding(i, winrate, score))

    print(f"   Generated {len(embeddings)} dummy embeddings")

    # Test different index types
    print("\n2. Building ANN indexes...")

    # HNSW - Approximate search (default, recommended)
    print("\n   a) HNSW Index (Approximate, Fast):")
    store_hnsw = PositionEmbeddingStore(
        index_type='HNSW',
        M=32,
        ef_construction=200,
        ef_search=128
    )
    store_hnsw.add_embeddings(embeddings)
    print(f"      Index built with {store_hnsw.size()} embeddings")
    print(f"      Embedding dimension: {store_hnsw.dimension}")

    # Flat - Exact search for comparison
    print("\n   b) Flat Index (Exact, Slower):")
    store_flat = PositionEmbeddingStore(index_type='Flat')
    store_flat.add_embeddings(embeddings)
    print(f"      Index built with {store_flat.size()} embeddings")

    # Use HNSW for the rest of the tests
    store = store_hnsw

    # Generate a query embedding that should be similar to a known embedding
    print("\n3. Creating query embedding...")
    target_idx = 420  # We'll make a query similar to embedding #420
    target_embedding = embeddings[target_idx]

    # Create a query that's 95% similar to the target
    query = generate_similar_embedding(target_embedding, idx=9999, similarity=0.95)

    print(f"   Query created (similar to embedding #{target_idx})")
    print(f"   Query winrate: {query['winrate']:.4f}, score: {query['score_lead']:.2f}")
    print(f"   Target winrate: {target_embedding['winrate']:.4f}, "
          f"score: {target_embedding['score_lead']:.2f}")

    # Search for nearest neighbors with both methods
    print("\n4. Searching for k=5 nearest neighbors...")
    k = 5

    print("\n   a) HNSW (Approximate) Results:")
    results_hnsw = store_hnsw.search(query, k=k)
    print("   " + "-" * 66)
    for rank, (embedding, distance) in enumerate(results_hnsw, 1):
        is_target = "*** TARGET ***" if embedding['query_id'] == target_embedding['query_id'] else ""
        print(f"   {rank}. {embedding['query_id']:20s} | Distance: {distance:10.4f} {is_target}")
        print(f"      Winrate: {embedding['winrate']:.4f} | Score: {embedding['score_lead']:6.2f}")
        if rank < k:
            print()

    print("\n   b) Flat (Exact) Results:")
    results_flat = store_flat.search(query, k=k)
    print("   " + "-" * 66)
    for rank, (embedding, distance) in enumerate(results_flat, 1):
        is_target = "*** TARGET ***" if embedding['query_id'] == target_embedding['query_id'] else ""
        print(f"   {rank}. {embedding['query_id']:20s} | Distance: {distance:10.4f} {is_target}")
        print(f"      Winrate: {embedding['winrate']:.4f} | Score: {embedding['score_lead']:6.2f}")
        if rank < k:
            print()

    # Compare results
    hnsw_ids = [emb['query_id'] for emb, _ in results_hnsw]
    flat_ids = [emb['query_id'] for emb, _ in results_flat]
    matches = sum(1 for id in hnsw_ids if id in flat_ids)
    print(f"\n   Overlap between HNSW and Exact: {matches}/{k} ({100*matches/k:.0f}%)")

    # Verify the target is in top results
    print("\n5. Verification:")
    found_in_top_hnsw = any(emb['query_id'] == target_embedding['query_id']
                            for emb, _ in results_hnsw)
    found_in_top_flat = any(emb['query_id'] == target_embedding['query_id']
                            for emb, _ in results_flat)

    print("   HNSW Index:")
    if found_in_top_hnsw:
        rank = next(i for i, (emb, _) in enumerate(results_hnsw, 1)
                   if emb['query_id'] == target_embedding['query_id'])
        print(f"   ✓ Target embedding found at rank {rank}/{k}")
    else:
        print(f"   ✗ Target embedding NOT in top {k}")

    print("   Flat Index:")
    if found_in_top_flat:
        rank = next(i for i, (emb, _) in enumerate(results_flat, 1)
                   if emb['query_id'] == target_embedding['query_id'])
        print(f"   ✓ Target embedding found at rank {rank}/{k}")
    else:
        print(f"   ✗ Target embedding NOT in top {k}")

    # Test exact hash lookup
    print("\n6. Testing exact sym_hash lookup...")
    exact_match = store.search_by_sym_hash(embeddings[100]['sym_hash'])
    if exact_match:
        print(f"   ✓ Found embedding by sym_hash: {exact_match['query_id']}")
    else:
        print("   ✗ Exact lookup failed")

    # Additional test: Find nearest neighbor to an existing embedding
    print("\n7. Testing self-similarity (query with existing embedding)...")
    self_query = embeddings[250]
    self_results = store.search(self_query, k=3)

    print(f"   Query: {self_query['query_id']}")
    print(f"   Top 3 matches:")
    for rank, (embedding, distance) in enumerate(self_results, 1):
        match_indicator = "SELF" if embedding['query_id'] == self_query['query_id'] else ""
        print(f"      {rank}. {embedding['query_id']:20s} | Distance: {distance:10.6f} {match_indicator}")

    # The first result should be itself with distance ~0
    if self_results[0][1] < 0.01:  # Very small distance threshold
        print(f"   ✓ Self-match verified (distance: {self_results[0][1]:.6f})")
    else:
        print(f"   ✗ Self-match failed (distance: {self_results[0][1]:.6f})")

    print("\n" + "=" * 70)
    print("Test completed successfully!")
    print("=" * 70)
    print("\nSummary:")
    print(f"  - HNSW provides fast approximate search")
    print(f"  - Tested with {num_embeddings} embeddings in 725 dimensions")
    print(f"  - HNSW accuracy: {100*matches/k:.0f}% match with exact search in top-{k}")
    print("=" * 70)


if __name__ == "__main__":
    main()
