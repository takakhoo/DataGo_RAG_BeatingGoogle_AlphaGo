"""
Quick test script for Phase 1b with minimal grid search
"""
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from tuning.phase1.phase1b_relevance_weights import Phase1bTuner, RelevanceWeights

def quick_test():
    """Run Phase 1b with minimal configuration for testing"""
    
    db_path = "data/tmp/test_phase1b_small.json"
    output_dir = "tuning_results/phase1b_test_quick"
    
    print("="*80)
    print("QUICK TEST: Phase 1b Relevance Weight Tuning")
    print("="*80)
    
    # Create tuner
    tuner = Phase1bTuner(
        rag_database_path=db_path,
        output_dir=output_dir,
        min_positions_per_group=3
    )
    
    print("\n" + "="*80)
    print("Testing baseline weights...")
    print("="*80)
    
    # Test baseline weights from claude_instructions.txt
    baseline_weights = RelevanceWeights(
        policy=0.40,
        winrate=0.25,
        score_lead=0.10,
        visit_distribution=0.15,
        stone_count=0.05,
        komi=0.05
    )
    
    metrics = tuner.evaluate_weights(baseline_weights)
    
    print(f"\nBaseline Results:")
    print(f"  Combined correlation: {metrics.combined_correlation:.4f}")
    print(f"  Policy stability correlation: {metrics.policy_stability_correlation:.4f}")
    print(f"  Value stability correlation: {metrics.value_stability_correlation:.4f}")
    print(f"  Num groups: {metrics.num_groups}")
    print(f"  Num comparisons: {metrics.num_comparisons}")
    
    print("\n" + "="*80)
    print("Testing alternative weight combinations...")
    print("="*80)
    
    # Test a few alternatives
    test_configs = [
        {'name': 'High Policy', 'weights': RelevanceWeights(0.50, 0.20, 0.10, 0.10, 0.05, 0.05)},
        {'name': 'High Winrate', 'weights': RelevanceWeights(0.30, 0.35, 0.10, 0.15, 0.05, 0.05)},
        {'name': 'Balanced', 'weights': RelevanceWeights(0.35, 0.25, 0.15, 0.15, 0.05, 0.05)},
    ]
    
    best_config = None
    best_correlation = metrics.combined_correlation
    
    for config in test_configs:
        test_metrics = tuner.evaluate_weights(config['weights'])
        print(f"\n{config['name']}:")
        print(f"  Combined correlation: {test_metrics.combined_correlation:.4f}")
        
        if test_metrics.combined_correlation > best_correlation:
            best_correlation = test_metrics.combined_correlation
            best_config = config
    
    print("\n" + "="*80)
    print("QUICK TEST COMPLETE")
    print("="*80)
    
    if best_config:
        print(f"\nBest configuration: {best_config['name']}")
        print(f"  Correlation: {best_correlation:.4f}")
    else:
        print(f"\nBaseline is best with correlation: {best_correlation:.4f}")
    
    print("\n✓ All components working correctly!")
    print(f"✓ Processed {metrics.num_groups} position groups")
    print(f"✓ Made {metrics.num_comparisons} pairwise comparisons")
    
    return True

if __name__ == "__main__":
    try:
        success = quick_test()
        if success:
            print("\n" + "="*80)
            print("SUCCESS: Phase 1b code is working correctly!")
            print("="*80)
            print("\nYou can now run the full training with:")
            print("  python tuning/phase1/phase1b_relevance_weights.py \\")
            print("    --rag-database data/tmp/test_phase1b_small.json \\")
            print("    --output-dir ./tuning_results/phase1b_test \\")
            print("    --method grid")
    except Exception as e:
        print(f"\n{'='*80}")
        print("ERROR during test:")
        print("="*80)
        print(f"{e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
