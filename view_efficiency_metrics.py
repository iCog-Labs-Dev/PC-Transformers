#!/usr/bin/env python3
"""
Script to view and compare efficiency metrics saved during training/evaluation.

Usage:
    python view_efficiency_metrics.py
"""

import json
import os
import glob
from datetime import datetime
from utils.pc_utils import load_and_compare_metrics


def print_metrics_summary():
    """Print a summary of all saved efficiency metrics."""
    print("=" * 80)
    print("EFFICIENCY METRICS SUMMARY")
    print("=" * 80)
    
    comparison = load_and_compare_metrics()
    
    if not comparison:
        print("No efficiency metrics found. Run training or evaluation first.")
        return
    
    # Print summary for each phase
    for phase, data in comparison.items():
        print(f"\n{phase.upper()} PHASE:")
        print(f"  Measurements: {data['count']}")
        print(f"  Average Memory Usage: {data['avg_memory_mb']:.1f} MB")
        print(f"  Average Normalization FLOPS: {data['avg_normalization_flops']:,}")
        
        latest = data['latest_metrics']
        flops_info = latest.get('flops_breakdown', {}).get('model_info', {})
        
        if flops_info:
            print(f"  Model Configuration:")
            print(f"    LayerNorm layers: {flops_info.get('layernorm_layers', 0)}")
            print(f"    RMSNorm layers: {flops_info.get('rmsnorm_layers', 0)}")
            print(f"    Hidden size: {flops_info.get('hidden_size', 'Unknown')}")
            print(f"    Batch size: {flops_info.get('batch_size', 'Unknown')}")
            print(f"    Sequence length: {flops_info.get('seq_len', 'Unknown')}")
    
    # Calculate potential savings if using only RMSNorm
    print(f"\n{'='*80}")
    print("POTENTIAL RMSNORM BENEFITS:")
    print("="*80)
    
    for phase, data in comparison.items():
        latest = data['latest_metrics']
        flops_breakdown = latest.get('flops_breakdown', {})
        model_info = flops_breakdown.get('model_info', {})
        
        layernorm_layers = model_info.get('layernorm_layers', 0)
        rmsnorm_layers = model_info.get('rmsnorm_layers', 0)
        total_layers = layernorm_layers + rmsnorm_layers
        
        if total_layers > 0:
            print(f"\n{phase.upper()} Phase Analysis:")
            
            # Current FLOPS
            current_flops = flops_breakdown.get('total_normalization', 0)
            
            # Calculate what FLOPS would be with all RMSNorm
            if model_info.get('hidden_size') and model_info.get('batch_size') and model_info.get('seq_len'):
                hidden_size = model_info['hidden_size']
                batch_size = model_info['batch_size']
                seq_len = model_info['seq_len']
                
                # FLOPS if all layers were RMSNorm
                all_rmsnorm_flops = total_layers * batch_size * seq_len * (3 * hidden_size + 2)
                
                # FLOPS if all layers were LayerNorm  
                all_layernorm_flops = total_layers * batch_size * seq_len * (5 * hidden_size + 2)
                
                if all_layernorm_flops > 0:
                    savings_percent = ((all_layernorm_flops - all_rmsnorm_flops) / all_layernorm_flops) * 100
                    print(f"  Current normalization FLOPS: {current_flops:,}")
                    print(f"  All LayerNorm FLOPS: {all_layernorm_flops:,}")
                    print(f"  All RMSNorm FLOPS: {all_rmsnorm_flops:,}")
                    print(f"  Potential FLOPS savings: {savings_percent:.1f}%")
                    
                    # Memory estimation (rough)
                    # LayerNorm has bias parameters, RMSNorm doesn't
                    param_savings = layernorm_layers * hidden_size * 4  # 4 bytes per float32
                    print(f"  Potential parameter memory savings: {param_savings/1024:.1f} KB")


def print_detailed_metrics():
    """Print detailed metrics for each measurement."""
    metrics_dir = "efficiency_metrics"
    
    if not os.path.exists(metrics_dir):
        print(f"Metrics directory {metrics_dir} not found")
        return
    
    metrics_files = sorted(glob.glob(f"{metrics_dir}/*.json"))
    
    if not metrics_files:
        print("No metrics files found")
        return
    
    print("\n" + "=" * 80)
    print("DETAILED METRICS")
    print("=" * 80)
    
    for file_path in metrics_files:
        try:
            with open(file_path, 'r') as f:
                metrics = json.load(f)
            
            filename = os.path.basename(file_path)
            phase = metrics.get('phase', 'unknown')
            timestamp = metrics.get('timestamp', 'unknown')
            epoch = metrics.get('epoch')
            
            print(f"\nFile: {filename}")
            print(f"Phase: {phase}")
            print(f"Timestamp: {timestamp}")
            if epoch is not None:
                print(f"Epoch: {epoch}")
            
            print(f"Memory Allocated: {metrics.get('memory_allocated_mb', 0):.1f} MB")
            print(f"Memory Cached: {metrics.get('memory_cached_mb', 0):.1f} MB")
            
            flops_breakdown = metrics.get('flops_breakdown', {})
            if 'total_normalization' in flops_breakdown:
                print(f"Normalization FLOPS: {flops_breakdown['total_normalization']:,}")
            
            if 'layernorm' in flops_breakdown:
                print(f"LayerNorm FLOPS: {flops_breakdown['layernorm']:,}")
            
            if 'rmsnorm' in flops_breakdown:
                print(f"RMSNorm FLOPS: {flops_breakdown['rmsnorm']:,}")
                
        except Exception as e:
            print(f"Error reading {file_path}: {e}")


def main():
    """Main function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="View efficiency metrics")
    parser.add_argument("--detailed", action="store_true", help="Show detailed metrics for each file")
    parser.add_argument("--clean", action="store_true", help="Remove all metrics files")
    
    args = parser.parse_args()
    
    if args.clean:
        metrics_dir = "efficiency_metrics"
        if os.path.exists(metrics_dir):
            import shutil
            shutil.rmtree(metrics_dir)
            print(f"Removed {metrics_dir} directory")
        else:
            print(f"Directory {metrics_dir} does not exist")
        return
    
    print_metrics_summary()
    
    if args.detailed:
        print_detailed_metrics()
    
    print(f"\n{'='*80}")
    print("NOTES:")
    print("- Metrics are automatically saved during training and evaluation")
    print("- Use --detailed flag to see individual measurements")
    print("- Use --clean flag to remove all saved metrics")
    print("- FLOPS calculations are estimates for normalization layers only")
    print("="*80)


if __name__ == "__main__":
    main()
