
import matplotlib.pyplot as plt
from pathlib import Path
from matplotlib.ticker import MaxNLocator

def plot_metrics(
    train_energies, 
    val_energies=None, 
    train_perplexities=None,
    val_perplexities=None
):
    """
    Plot training and validation metrics.
    """
    assets_dir = Path("assets")
    assets_dir.mkdir(exist_ok=True)

    # Energy Plot
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

    #  Perplexity Plot
    if train_perplexities is not None:
        plt.figure(figsize=(10, 5))
        epochs = range(1, len(train_perplexities) + 1)

        plt.plot(epochs, train_perplexities, 'b-', label='Training')
        if val_perplexities:
            plt.plot(epochs, val_perplexities, 'r-', label='Validation')

        plt.title('Training/Validation Perplexity vs. Epochs')
        plt.xlabel('Epoch')
        plt.ylabel('Perplexity')
        plt.legend()
        plt.grid(True)

        ax = plt.gca()
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))

        save_path_ppl = assets_dir / 'perplexity_plot.png'
        plt.savefig(save_path_ppl)
        plt.close()
        print(f"Perplexity plot saved to: {save_path_ppl}")