"""
Visualization Script for Predictive Coding Transformer

This script provides utilities for plotting training and validation metrics.

Usage:
    python visualization.py

Example:
    python visualization.py
"""
import matplotlib.pyplot as plt
from pathlib import Path
from matplotlib.ticker import MaxNLocator

def plot_metrics(train_energies, val_energies):
    """
    Plot training and validation energy metrics over epochs.

    Args:
        train_energies (list of float): List of training energy values per epoch.
        val_energies (list of float): List of validation energy values per epoch.

    Returns:
        None
    """
    assets_dir = Path("assets")
    assets_dir.mkdir(exist_ok=True)

    plt.figure(figsize=(10, 5))
    epochs = range(1, len(train_energies) + 1)
    
    plt.plot(epochs, train_energies, 'b-', label='Training')
    if val_energies:
        plt.plot(epochs, val_energies, 'r-', label='Validation')
    
    plt.title('Training/Validation Energy vs. Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Energy')
    plt.legend()
    plt.grid(True)
    
    ax = plt.gca()
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    
    save_path = assets_dir / 'energy_plot.png'
    plt.savefig(save_path)
    plt.close()
    print(f"Energy plot saved to: {save_path}")
