import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

def plot_embeddings(embeddings, labels, title='Embeddings'):
    """Plot embeddings with t-SNE"""
    tsne = TSNE(n_components=2, random_state=42)
    embeddings_tsne = tsne.fit_transform(embeddings)
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(embeddings_tsne[:, 0], embeddings_tsne[:, 1], c=labels, cmap='viridis', alpha=0.6)
    plt.colorbar(scatter)
    plt.title(title)
    plt.xlabel('t-SNE Axis 1')
    plt.ylabel('t-SNE Axis 2')
    plt.grid(True)
    plt.savefig('results/figures/embeddings_tsne.png')
    plt.show()
