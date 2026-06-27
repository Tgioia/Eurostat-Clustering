import pandas as pd
import ollama
import os

def get_cluster_name(terms_sample, cluster_id, used_names):

    forbidden_names = ", ".join(used_names) if used_names else "None yet"
    
    prompt = f"""You are an expert statistical taxonomist. 
I have grouped a set of statistical measures into a cluster.
Based on the following sample of terms, provide a single, short, and accurate domain name (e.g., "Agriculture", "Transport", "Economy").

CRITICAL RULES:
1. Output ONLY the domain name. Do not add any punctuation, quotes, or conversational text.
2. UNIQUENESS: You MUST NOT use any of the following names, as they are already assigned to other clusters: [{forbidden_names}]
3. If the terms seem similar to a used name, find a more specific sub-category name.

TERMS: {', '.join(terms_sample)}
"""
    try:
        response = ollama.chat(
            model='ibm/granite4.1:3b',
            messages=[{'role': 'user', 'content': prompt}],
            options={'temperature': 0.1} 
        )

        name = response['message']['content'].strip().strip('"').strip("'")
        return name
    except Exception as e:
        print(f"Error asking LLM for cluster {cluster_id}: {e}")
        return f"Domain {cluster_id}"

def main():
    input_file = 'outputs/clustered_measures.csv'
    output_file = 'outputs/final_named_measures.csv'
    
    if not os.path.exists(input_file):
        print(f"Error: {input_file} not found. Make sure you ran the clustering script first.")
        return

    print(f"Loading {input_file}...")
    df = pd.read_csv(input_file)
    
    cluster_col = 'cluster' if 'cluster' in df.columns else 'cluster_id'
    
    if cluster_col not in df.columns:
        print(f"Error: No cluster column found in the CSV.")
        return

    unique_clusters = sorted(df[cluster_col].unique())
    print(f"Found {len(unique_clusters)} clusters. Booting up Granite to name them...\n")
    
    domain_mapping = {}
    used_names = []
    
    for cluster_id in unique_clusters:
        cluster_terms = df[df[cluster_col] == cluster_id]['term'].dropna().tolist()
        sample_for_llm = cluster_terms[:30] 
        
        preview = ", ".join(sample_for_llm[:5])
        print(f"Analyzing Cluster {cluster_id} (Sample: {preview}...)")
        
        cluster_name = get_cluster_name(sample_for_llm, cluster_id, used_names)
        
        domain_mapping[cluster_id] = cluster_name
        used_names.append(cluster_name)
        
        print(f" -> LLM Name: [{cluster_name}]\n")


    df['domain_name'] = df[cluster_col].map(domain_mapping)

    df.to_csv(output_file, index=False)
    print(f"[SUCCESS] Saved final named domains to {output_file}")

if __name__ == "__main__":
    main()