import pandas as pd
import numpy as np
import os
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity

def main():
    input_file = 'outputs/measures.csv'
    
    if not os.path.exists(input_file):
        print(f"Error: {input_file} not found. Run the classification script first.")
        return

    # 1. Load the measures
    print("Loading measures...")
    df = pd.read_csv(input_file)
    terms = df['term'].dropna().tolist()
    print(f"Loaded {len(terms)} measures.")

    # 2. Generate Embeddings (Transforms words into numbers)
    print("\nLoading NLP Model (this may take a moment the first time)...")
    # 'all-MiniLM-L6-v2' is a very fast and efficient pre-trained model for English
    model = SentenceTransformer('all-MiniLM-L6-v2') 
    
    print("Generating mathematical vectors (embeddings) for all terms...")
    embeddings = model.encode(terms)

    # 3. STEP 7: Unsupervised Clustering (Grouping into Domains)
    num_clusters = 9 # We guess 8 domains, similar to Eurostat's actual structure
    print(f"\nPerforming K-Means clustering to find {num_clusters} semantic domains...")
    
    kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(embeddings)

    # Save the clustered data
    df['cluster'] = cluster_labels
    output_cluster_file = 'outputs/clustered_measures.csv'
    df.to_csv(output_cluster_file, index=False)
    print(f"Saved clustered domains to {output_cluster_file}")

    # Display a sample from each cluster so you can name them later
    print("\n--- CLUSTER SAMPLES ---")
    for i in range(num_clusters):
        cluster_terms = df[df['cluster'] == i]['term'].tolist()
        # Print the first 10 terms of each cluster
        print(f"Cluster {i}: {', '.join(cluster_terms[:10])}")

    # 4. STEP 8: Semantic Relationships (Cosine Similarity)
    print("\n--- SEMANTIC RELATIONSHIPS (For Step 8) ---")
    print("Calculating the similarity matrix for all terms...")
    similarity_matrix = cosine_similarity(embeddings)

    # Let's find the closest semantic matches for a random test word (e.g., the 10th word in your list)
    test_index = 10 
    if test_index < len(terms):
        target_term = terms[test_index]
        print(f"\nFinding terms most semantically related to: '{target_term}'")
        
        # Get similarities for this specific term
        similarities = similarity_matrix[test_index]
        
        # Sort indexes by highest similarity (skipping the first one, which is the word itself)
        most_similar_indexes = np.argsort(similarities)[::-1][1:6]
        
        for idx in most_similar_indexes:
            score = similarities[idx] * 100
            match_term = terms[idx]
            print(f" -> {match_term} (Similarity: {score:.1f}%)")

if __name__ == "__main__":
    main()