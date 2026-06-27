import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import os

def main():
    input_file = 'outputs/measures.csv'
    output_image = 'outputs/cluster_plot.png'
    
    if not os.path.exists(input_file):
        print(f"Error: {input_file} not found.")
        return

    print("Loading data and generating vectors...")
    df = pd.read_csv(input_file)
    terms = df['term'].dropna().tolist()

    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = model.encode(terms)

    print("Clustering...")
    kmeans = KMeans(n_clusters=8, init='k-means++', random_state=42, n_init=10)
    df['cluster'] = kmeans.fit_predict(embeddings)

    print("Reducing 384 dimensions to 2D using PCA...")
    pca = PCA(n_components=2, random_state=42)
    reduced_vectors = pca.fit_transform(embeddings)
    
    # --- ADD THESE 3 LINES ---
    variance = pca.explained_variance_ratio_
    total_variance = sum(variance) * 100
    print(f"\nExplained Variance -> PC1: {variance[0]*100:.1f}%, PC2: {variance[1]*100:.1f}%")
    print(f"Total Variance Captured in 2D Plot: {total_variance:.1f}%\n")
    # -------------------------

    # Add the X and Y coordinates to our dataframe
    df['x'] = reduced_vectors[:, 0]
    df['y'] = reduced_vectors[:, 1]

    print("Drawing the plot...")
    plt.figure(figsize=(10, 7))
    sns.scatterplot(
        data=df, 
        x='x', 
        y='y', 
        hue='cluster', 
        palette='Set2', 
        s=60, # Dot size
        alpha=0.8, # Slight transparency
        edgecolor=None
    )

    plt.title("2D PCA Projection of Statistical Measure Clusters", fontsize=14, pad=15)
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.legend(title='Cluster ID', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()

    plt.savefig(output_image, dpi=300) # Save as high-resolution PNG
    print(f"[SUCCESS] Saved plot to {output_image}")

if __name__ == "__main__":
    main()