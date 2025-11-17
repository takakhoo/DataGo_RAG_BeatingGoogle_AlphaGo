"""
Minimal test for phase1_uncertainty_tuning.py
Tests with a tiny subset of data to verify functionality
"""

import json
import numpy as np
from pathlib import Path

# Create minimal test data
def create_minimal_test_data():
    """Create 2 tiny JSON files with just 4-5 positions each"""
    output_dir = Path("./test_data_minimal")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create 5 test positions with matched sym_hash
    test_positions = []
    for i in range(5):
        sym_hash = f"test_sym_hash_{i}"
        stones = 30 + i * 30  # Vary from 30 to 150 stones
        
        # Create shallow policy (more uncertain)
        shallow_policy = np.random.dirichlet(np.ones(361) * 2)
        
        # Create deep policy (more certain - peaked)
        deep_policy = np.random.dirichlet(np.ones(361) * 0.5)
        
        # Shallow entry
        shallow_entry = {
            "sym_hash": sym_hash,
            "state_hash": f"state_{i}",
            "policy": shallow_policy.tolist() + [0.0],  # 362 total (361 + pass)
            "ownership": np.random.rand(361).tolist(),
            "winrate": 0.5 + np.random.randn() * 0.1,
            "score_lead": np.random.randn() * 3,
            "stone_count": stones,
            "komi": 7.5,
            "query_id": f"query_{i}",
            "child_nodes": [
                {"move": j, "visits": 100-j*10, "value": 0.5 + np.random.randn()*0.1}
                for j in range(5)
            ]
        }
        
        # Deep entry (similar but with variations)
        deep_entry = {
            "sym_hash": sym_hash,
            "state_hash": f"state_{i}",
            "policy": deep_policy.tolist() + [0.0],
            "ownership": np.random.rand(361).tolist(),
            "winrate": shallow_entry["winrate"] + np.random.randn() * 0.05,
            "score_lead": shallow_entry["score_lead"] + np.random.randn() * 1,
            "stone_count": stones,
            "komi": 7.5,
            "query_id": f"query_{i}",
            "child_nodes": [
                {"move": j, "visits": 500-j*50, "value": 0.5 + np.random.randn()*0.05}
                for j in range(8)
            ]
        }
        
        test_positions.append((shallow_entry, deep_entry))
    
    # Save shallow and deep databases
    shallow_db = [pos[0] for pos in test_positions]
    deep_db = [pos[1] for pos in test_positions]
    
    shallow_path = output_dir / "shallow_test.json"
    deep_path = output_dir / "deep_test.json"
    
    with open(shallow_path, 'w') as f:
        json.dump(shallow_db, f, indent=2)
    
    with open(deep_path, 'w') as f:
        json.dump(deep_db, f, indent=2)
    
    print(f"✓ Created test data:")
    print(f"  Shallow DB: {shallow_path} ({len(shallow_db)} positions)")
    print(f"  Deep DB: {deep_path} ({len(deep_db)} positions)")
    
    return str(shallow_path), str(deep_path)


def test_phase1_tuning():
    """Run minimal test of Phase1Tuner"""
    print("\n" + "="*80)
    print("MINIMAL TEST: phase1_uncertainty_tuning.py")
    print("="*80)
    
    # Import the module and disable W&B for testing
    import phase1_uncertainty_tuning
    phase1_uncertainty_tuning.WANDB_AVAILABLE = False  # Disable W&B for test
    from phase1_uncertainty_tuning import Phase1Tuner, UncertaintyConfig
    
    # Create test data
    print("\n1. Creating minimal test data...")
    shallow_path, deep_path = create_minimal_test_data()
    
    # Initialize tuner
    print("\n2. Initializing Phase1Tuner...")
    tuner = Phase1Tuner(
        shallow_db_path=shallow_path,
        deep_db_path=deep_path,
        output_dir="./test_results_minimal",
        train_split=0.8
    )
    
    print(f"\n✓ Tuner initialized successfully!")
    print(f"  Training positions: {len(tuner.positions_train)}")
    print(f"  Validation positions: {len(tuner.positions_val)}")
    
    # Test a single configuration
    print("\n3. Testing a single uncertainty configuration...")
    test_config = UncertaintyConfig(
        w1=0.5,
        w2=0.5,
        phase_function_type='linear',
        phase_coefficients=[0.0, 1.0]
    )
    
    results = tuner.evaluate_config(test_config, tuner.positions_train)
    print(f"\n✓ Configuration evaluated successfully!")
    print(f"  Pearson correlation: {results['pearson_correlation']:.4f}")
    print(f"  Top-K precision: {results['top_k_precision']:.4f}")
    print(f"  Mean uncertainty: {results['mean_uncertainty']:.4f}")
    
    # Test grid search with just 2 configs
    print("\n4. Testing mini grid search (2 configs)...")
    
    # Override generate_configs to return just 2 configs for quick test
    original_generate = tuner.generate_configs
    def mini_generate():
        return [
            UncertaintyConfig(w1=0.4, w2=0.6, phase_function_type='linear', 
                            phase_coefficients=[0.0, 1.0]),
            UncertaintyConfig(w1=0.6, w2=0.4, phase_function_type='linear', 
                            phase_coefficients=[0.3, 0.85])
        ]
    
    tuner.generate_configs = mini_generate
    best_config, _ = tuner.grid_search()
    tuner.generate_configs = original_generate  # Restore
    
    print(f"\n✓ Grid search completed!")
    print(f"  Best w1: {best_config.w1:.4f}")
    print(f"  Best w2: {best_config.w2:.4f}")
    
    # Test threshold analysis
    print("\n5. Testing threshold analysis...")
    threshold_analysis = tuner.find_storage_threshold(
        best_config,
        target_percentiles=[10, 20]
    )
    
    print(f"\n✓ Threshold analysis completed!")
    print(f"  Analyzed {len(threshold_analysis)} thresholds")
    
    print("\n" + "="*80)
    print("✓ ALL TESTS PASSED!")
    print("="*80)
    print("\nYour phase1_uncertainty_tuning.py is working correctly!")
    print("You can now run it with real data using:")
    print("  python phase1_uncertainty_tuning.py --shallow-db <path> --deep-db <path>")
    print("="*80)


if __name__ == "__main__":
    test_phase1_tuning()
