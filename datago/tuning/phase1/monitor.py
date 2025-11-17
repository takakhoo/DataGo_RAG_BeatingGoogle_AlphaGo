"""
Monitoring Dashboard for Phase 1 Tuning
Real-time monitoring of tuning progress, GPU utilization, and results

Run this alongside the tuning scripts to monitor progress.
"""

import os
import json
import time
import psutil
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import subprocess


class TuningMonitor:
    """Monitor tuning progress and system resources"""
    
    def __init__(self, results_dir: str = "./tuning_results"):
        self.results_dir = Path(results_dir)
        self.start_time = time.time()
        
    def get_gpu_stats(self) -> Dict:
        """Get GPU utilization and memory stats using nvidia-smi"""
        try:
            result = subprocess.run(
                ['nvidia-smi', '--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu',
                 '--format=csv,noheader,nounits'],
                capture_output=True,
                text=True,
                timeout=5
            )
            
            if result.returncode == 0:
                values = result.stdout.strip().split(',')
                return {
                    'gpu_utilization': float(values[0]),
                    'memory_used_mb': float(values[1]),
                    'memory_total_mb': float(values[2]),
                    'temperature_c': float(values[3]),
                    'memory_utilization': (float(values[1]) / float(values[2])) * 100
                }
        except Exception as e:
            print(f"Warning: Could not get GPU stats: {e}")
        
        return {}
    
    def get_cpu_stats(self) -> Dict:
        """Get CPU and RAM statistics"""
        return {
            'cpu_percent': psutil.cpu_percent(interval=1),
            'ram_percent': psutil.virtual_memory().percent,
            'ram_used_gb': psutil.virtual_memory().used / (1024**3),
            'ram_total_gb': psutil.virtual_memory().total / (1024**3)
        }
    
    def get_disk_stats(self, path: str = ".") -> Dict:
        """Get disk usage statistics"""
        usage = psutil.disk_usage(path)
        return {
            'disk_used_gb': usage.used / (1024**3),
            'disk_total_gb': usage.total / (1024**3),
            'disk_percent': usage.percent
        }
    
    def load_latest_results(self, phase: str = "phase1") -> Optional[Dict]:
        """Load the most recent results from a phase"""
        results_file = self.results_dir / phase / f"{phase}_results.json"
        
        if not results_file.exists():
            return None
        
        try:
            with open(results_file, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"Warning: Could not load results: {e}")
            return None
    
    def print_status(self, phase: str = "phase1"):
        """Print current status"""
        os.system('clear' if os.name != 'nt' else 'cls')
        
        print("="*80)
        print(f"RAG-ALPHAGO TUNING MONITOR - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("="*80)
        
        # Elapsed time
        elapsed = time.time() - self.start_time
        hours = int(elapsed // 3600)
        minutes = int((elapsed % 3600) // 60)
        print(f"\nElapsed Time: {hours}h {minutes}m")
        
        # System stats
        print("\n" + "-"*80)
        print("SYSTEM RESOURCES")
        print("-"*80)
        
        gpu_stats = self.get_gpu_stats()
        if gpu_stats:
            print(f"GPU Utilization:  {gpu_stats['gpu_utilization']:>5.1f}%  ", end="")
            self._print_bar(gpu_stats['gpu_utilization'], target=90)
            print(f"GPU Memory:       {gpu_stats['memory_used_mb']:>8.0f} MB / {gpu_stats['memory_total_mb']:.0f} MB ({gpu_stats['memory_utilization']:.1f}%)")
            print(f"GPU Temperature:  {gpu_stats['temperature_c']:>5.0f}°C")
        else:
            print("GPU Stats: Not available")
        
        cpu_stats = self.get_cpu_stats()
        print(f"\nCPU Utilization:  {cpu_stats['cpu_percent']:>5.1f}%")
        print(f"RAM Usage:        {cpu_stats['ram_used_gb']:>5.1f} GB / {cpu_stats['ram_total_gb']:.1f} GB ({cpu_stats['ram_percent']:.1f}%)")
        
        disk_stats = self.get_disk_stats()
        print(f"Disk Usage:       {disk_stats['disk_used_gb']:>5.1f} GB / {disk_stats['disk_total_gb']:.1f} GB ({disk_stats['disk_percent']:.1f}%)")
        
        # Tuning progress
        print("\n" + "-"*80)
        print(f"TUNING PROGRESS - {phase.upper()}")
        print("-"*80)
        
        results = self.load_latest_results(phase)
        if results:
            if isinstance(results, list):
                # Phase 1 format: list of config results
                total_configs = len(results)
                if total_configs > 0:
                    print(f"Configurations tested: {total_configs}")
                    
                    # Show recent configs
                    print("\nRecent Results:")
                    for i, result in enumerate(results[-3:], 1):
                        config_id = result.get('config', {}).get('config_id', 'unknown')
                        win_rate = result.get('win_rate', 0)
                        games = result.get('total_games', 0)
                        print(f"  {i}. {config_id[:50]:<50} | WR: {win_rate:.3f} ({games} games)")
                    
                    # Best so far
                    best = max(results, key=lambda x: x.get('win_rate', 0))
                    print(f"\nBest Win Rate So Far: {best.get('win_rate', 0):.3f}")
                    print(f"  Config: {best.get('config', {}).get('config_id', 'unknown')}")
                else:
                    print("No results yet...")
            else:
                # Single result format
                print(f"Win Rate: {results.get('win_rate', 'N/A')}")
                print(f"Games: {results.get('total_games', 'N/A')}")
        else:
            print("Waiting for results...")
        
        # Performance warnings
        print("\n" + "-"*80)
        print("STATUS")
        print("-"*80)
        
        warnings = []
        if gpu_stats and gpu_stats['gpu_utilization'] < 80:
            warnings.append(f"⚠️  Low GPU utilization ({gpu_stats['gpu_utilization']:.1f}%) - pipeline bottleneck?")
        if gpu_stats and gpu_stats['temperature_c'] > 80:
            warnings.append(f"⚠️  High GPU temperature ({gpu_stats['temperature_c']:.0f}°C)")
        if cpu_stats['ram_percent'] > 90:
            warnings.append(f"⚠️  High RAM usage ({cpu_stats['ram_percent']:.1f}%)")
        if disk_stats['disk_percent'] > 90:
            warnings.append(f"⚠️  Low disk space ({disk_stats['disk_percent']:.1f}% used)")
        
        if warnings:
            for warning in warnings:
                print(warning)
        else:
            print("✓ All systems normal")
        
        print("\n" + "="*80)
        print("Press Ctrl+C to stop monitoring")
        print("="*80)
    
    def _print_bar(self, value: float, width: int = 30, target: float = 100):
        """Print a progress bar"""
        filled = int((value / target) * width)
        bar = "█" * filled + "░" * (width - filled)
        print(f"[{bar}]")
    
    def run(self, phase: str = "phase1", refresh_interval: int = 10):
        """Run monitoring loop"""
        print("Starting monitoring...")
        print(f"Monitoring directory: {self.results_dir}")
        print(f"Refresh interval: {refresh_interval} seconds")
        print("\n")
        
        try:
            while True:
                self.print_status(phase)
                time.sleep(refresh_interval)
        except KeyboardInterrupt:
            print("\n\nMonitoring stopped.")


class QuickStats:
    """Quick statistics calculator for tuning results"""
    
    @staticmethod
    def summarize_phase1(results_file: str):
        """Summarize Phase 1 results"""
        with open(results_file, 'r') as f:
            results = json.load(f)
        
        if not results:
            print("No results found.")
            return
        
        print("="*80)
        print("PHASE 1 SUMMARY")
        print("="*80)
        
        # Overall stats
        total_configs = len(results)
        total_games = sum(r['total_games'] for r in results)
        
        print(f"\nTotal configurations tested: {total_configs}")
        print(f"Total games played: {total_games}")
        
        # Best configs
        sorted_results = sorted(results, key=lambda x: x['win_rate'], reverse=True)
        
        print("\nTop 5 Configurations:")
        print("-"*80)
        print(f"{'Rank':<6} {'Win Rate':<10} {'Games':<8} {'w1':<8} {'w2':<8} {'Early':<8} {'Late':<8}")
        print("-"*80)
        
        for i, result in enumerate(sorted_results[:5], 1):
            config = result['config']
            print(f"{i:<6} {result['win_rate']:<10.3f} {result['total_games']:<8} "
                  f"{config['w1']:<8.3f} {config['w2']:<8.3f} "
                  f"{config['phase_early_multiplier']:<8.3f} {config['phase_late_multiplier']:<8.3f}")
        
        # Parameter analysis
        print("\n" + "="*80)
        print("PARAMETER IMPACT ANALYSIS")
        print("="*80)
        
        # Group by w1 values
        from collections import defaultdict
        w1_groups = defaultdict(list)
        for r in results:
            w1 = r['config']['w1']
            w1_groups[w1].append(r['win_rate'])
        
        print("\nAverage Win Rate by w1 (policy entropy weight):")
        for w1 in sorted(w1_groups.keys()):
            avg_wr = sum(w1_groups[w1]) / len(w1_groups[w1])
            print(f"  w1={w1:.2f}: {avg_wr:.3f} (n={len(w1_groups[w1])})")


def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Monitor tuning progress")
    parser.add_argument("--mode", choices=["monitor", "summary"], default="monitor",
                       help="Mode: monitor (live) or summary (analyze results)")
    parser.add_argument("--results-dir", type=str, default="./tuning_results",
                       help="Results directory")
    parser.add_argument("--phase", type=str, default="phase1",
                       help="Phase to monitor")
    parser.add_argument("--refresh", type=int, default=10,
                       help="Refresh interval in seconds")
    parser.add_argument("--results-file", type=str,
                       help="Results file for summary mode")
    
    args = parser.parse_args()
    
    if args.mode == "monitor":
        monitor = TuningMonitor(results_dir=args.results_dir)
        monitor.run(phase=args.phase, refresh_interval=args.refresh)
    elif args.mode == "summary":
        if not args.results_file:
            print("Error: --results-file required for summary mode")
            return
        QuickStats.summarize_phase1(args.results_file)


if __name__ == "__main__":
    main()
