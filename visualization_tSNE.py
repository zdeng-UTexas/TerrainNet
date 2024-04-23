import numpy as np
import pandas as pd
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

def load_embeddings(csv_file):
    # Load embeddings with chunksize to handle large files more efficiently
    chunk_size = 1000  # Adjust this based on your available memory and file size
    chunks = pd.read_csv(csv_file, header=None, chunksize=chunk_size)
    data = pd.concat(chunks)  # Concatenate chunks into a single DataFrame
    paths = data.iloc[:, 0]
    embeddings = data.iloc[:, 1:].astype(np.float32)  # Ensure data type is float32 to reduce memory usage
    return paths, embeddings

def perform_tsne(embeddings, n_components=2, perplexity=30.0, learning_rate=200.0):
    tsne = TSNE(n_components=n_components, perplexity=perplexity, learning_rate=learning_rate)
    transformed_embeddings = tsne.fit_transform(embeddings)
    return transformed_embeddings

def plot_embeddings(transformed_embeddings, paths, plot_size=(12, 8), output_file='tsne_plot.png'):
    plt.figure(figsize=plot_size)
    plt.scatter(transformed_embeddings[:, 0], transformed_embeddings[:, 1], marker='o', s=10)  # Reduced marker size
    # Optional: Comment out the annotations if there are too many points, as it drastically slows down the plot
    # for i, path in enumerate(paths):
    #     plt.annotate(path.split('/')[-1], (transformed_embeddings[i, 0], transformed_embeddings[i, 1]))
    plt.title('t-SNE Visualization of Embedding Vectors')
    plt.xlabel('Component 1')
    plt.ylabel('Component 2')
    plt.grid(True)
    plt.savefig(output_file)
    plt.close()

# Example usage
csv_file = '../../experiment/embeddings_5_demo.csv'  # Path to your CSV file
paths, embeddings = load_embeddings(csv_file)
transformed_embeddings = perform_tsne(embeddings)
plot_embeddings(transformed_embeddings, paths)
