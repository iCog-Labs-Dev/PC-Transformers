import torch
import json
from pathlib import Path
import matplotlib.pyplot as plt
import torch.distributed as dist

from eval import evaluate
from training import train
from utils.device_utils import setup_device
from data_preparation.config import vocab_size
from predictive_coding.config import GPTConfig
from utils.config_utils import load_best_config
from data_preparation.dataloader import get_loaders
from model_architecture.pc_t_model import PCTransformer

def benchmark_blocks(block_values=[2, 3, 4, 5, 6], num_epochs=5):
    """Run controlled experiments varying only n_blocks"""
    local_rank, device, use_ddp = setup_device()
    
    if use_ddp and not dist.is_initialized():
        dist.init_process_group(backend="nccl")
    
    # Load best config
    best_config = load_best_config()
    results = []
    
    for n_blocks in block_values:
        if local_rank == 0:
            print(f"\n{'='*80}")
            print(f"Testing with {n_blocks} blocks")
            print(f"{'='*80}\n")
        
        # Create config with this n_blocks value
        config = GPTConfig(
            vocab_size = vocab_size,
            block_size = best_config["block_size"],
            lr = best_config["lr"],
            inference_lr = best_config["inference_lr"],
            peak_learning_rate = best_config["peak_learning_rate"],
            warmup_steps = best_config["warmup_steps"],
            n_embed = best_config["n_embed"],
            dropout = best_config["dropout"],
            T = best_config["T"],
            num_heads = best_config["num_heads"],
            n_blocks = n_blocks,   # Variable parameter
            batch_size = best_config["batch_size"],
            num_epochs = num_epochs,
            update_bias = best_config["update_bias"],
            internal_energy_fn_name = best_config["internal_energy_fn_name"],
            output_energy_fn_name = best_config["output_energy_fn_name"],
            combined_internal_weight = best_config["combined_internal_weight"],
            combined_output_weight = best_config["combined_output_weight"],
            use_flash_attention = best_config["use_flash_attention"],
            alpha = best_config["alpha"]
        )
        
        # Initialize model
        model = PCTransformer(config).to(device)
        if use_ddp:
            from torch.nn.parallel import DistributedDataParallel as DDP
            model = DDP(model, device_ids=[local_rank], output_device=local_rank)
        
        # Get data loaders
        train_loader, valid_loader, _ = get_loaders(distributed=use_ddp)
        
        # Train for specified epochs
        train_energies = []
        val_energies = []
        val_perplexities = []
        
        global_step = 0 
        
        for epoch in range(num_epochs):
            model.train()
            train_energy, train_perplexity, global_step = train(
                model, train_loader, config, global_step=global_step, device=device, logger=None
            )
            train_energies.append(train_energy)
            
            model.eval()
            with torch.no_grad():
                val_energy, val_perplexity = evaluate(
                    model, config, valid_loader, max_batches=None, device=device
                )
                
            val_energies.append(val_energy)
            val_perplexities.append(val_perplexity)
            
            if local_rank == 0:
                print(f"Epoch {epoch+1}/{num_epochs} | "
                      f"Train Energy: {train_energy:.4f} | "
                      f"Val Energy: {val_energy:.4f} | "
                      f"Val Perplexity: {val_perplexity:.4f}")
        
        # Store results
        if local_rank == 0:
            results.append({
                'n_blocks': n_blocks,
                'final_train_energy': train_energies[-1],
                'final_val_energy': val_energies[-1],
                'final_val_perplexity': val_perplexities[-1]
            })
        
        # Cleanup
        del model
        torch.cuda.empty_cache()
    
    # Save and plot results
    if local_rank == 0:
        # Save raw results
        output_dir = Path("assets/benchmarks")
        output_dir.mkdir(exist_ok=True)
        
        with open(output_dir / "blocks_benchmark.json", 'w') as f:
            json.dump(results, f, indent=2)
        
        # Create plots
        plot_blocks_benchmark(results, output_dir)
    
    if use_ddp:
        dist.destroy_process_group()

def plot_blocks_benchmark(results, output_dir):
    """Create benchmark plots"""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    n_blocks = [r['n_blocks'] for r in results]
    final_train_energy = [r['final_train_energy'] for r in results]
    final_val_energy = [r['final_val_energy'] for r in results]
    final_val_perplexity = [r['final_val_perplexity'] for r in results]
    
    # Plot 1: Final Training Energy
    axes[0].plot(n_blocks, final_train_energy, marker='o', linewidth=2)
    axes[0].set_xlabel('Number of Blocks')
    axes[0].set_ylabel('Final Training Energy')
    axes[0].set_title('Training Energy vs Number of Blocks')
    axes[0].grid(True, alpha=0.3)
    axes[0].set_xticks(n_blocks)
    
    # Plot 2: Final Validation Energy
    axes[1].plot(n_blocks, final_val_energy, marker='s', linewidth=2, color='orange')
    axes[1].set_xlabel('Number of Blocks')
    axes[1].set_ylabel('Final Validation Energy')
    axes[1].set_title('Validation Energy vs Number of Blocks')
    axes[1].grid(True, alpha=0.3)
    axes[1].set_xticks(n_blocks)
    
    # Plot 3: Final Perplexity
    axes[2].plot(n_blocks, final_val_perplexity, marker='^', linewidth=2, color='green')
    axes[2].set_xlabel('Number of Blocks')
    axes[2].set_ylabel('Validation Perplexity')
    axes[2].set_title('Perplexity vs Number of Blocks')
    axes[2].grid(True, alpha=0.3)
    axes[2].set_xticks(n_blocks)
    
    plt.tight_layout()
    plt.savefig(output_dir / "blocks_benchmark.png", dpi=300, bbox_inches='tight')
    print(f"Benchmark plot saved to {output_dir / 'blocks_benchmark.png'}")

if __name__ == "__main__":
    benchmark_blocks(block_values=[2, 3, 4, 5, 6], num_epochs=5)
