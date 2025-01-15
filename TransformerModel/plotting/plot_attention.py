import matplotlib.pyplot as plt
import seaborn as sns

def plot_attention(attention_weights, tokens, title='Attention Weights'):
    """Plot attention weights"""

    plt.figure(figsize=(10, 8))
    sns.heatmap(attention_weights, annot=True, fmt=".2f", xticklabels=tokens, yticklabels=tokens)
    plt.title(title)
    plt.xlabel('Keys')
    plt.ylabel('Queries')
    plt.savefig('results/figures/attention_weights.png')
    plt.show()
