import matplotlib.pyplot as plt

def plot_metrics(train_losses, val_losses, title='Model Loss'):
    """Plot training and validation loss"""
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title(title)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig('results/figures/training_validation_loss.png')
    plt.show()
