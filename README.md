# Eurostat Statistical Vocabulary Pipeline

This repository contains an automated, machine learning pipeline designed to extract, classify, and semantically structure Eurostat statistical vocabularies (based on the STAR paper dataset). 

Rather than relying on manual taxonomy creation, this project utilizes a privacy-preserving, local Large Language Model (LLM) alongside unsupervised machine learning to dynamically categorize complex database codes.

## Methodology & Architecture

1. **Data Extraction & Vocabulary Generation**
   * Raw Eurostat data and NUTS geographic tables are processed to build a comprehensive global vocabulary (`global_vocabulary.txt`).

2. **Deterministic Term Classification (Step 6)**
   * **Model:** `ibm/granite4.1:3b` running locally via Ollama.
   * **Execution:** Processed via batching with a strict temperature of `0.0` to guarantee deterministic JSON outputs and prevent hallucinations.
   * **Result:** Raw vocabularies were cleanly sorted into Measures, Dimension Names, Dimension Values, Units, and an explicit discard pile (`other_discarded.csv`).

3. **Unsupervised Semantic Structuring (Step 7)**
   * **Vectorization:** Raw alphanumeric database codes (e.g., `sdg 02 10`, `med en2`) were mapped to a 384-dimensional space using `all-MiniLM-L6-v2`.
   * **Clustering:** Partitioned into semantic domains using the **K-Means++** algorithm.
   * **State-Aware Naming:** The local Granite 3B model analyzed the clustered vectors and generated highly accurate domain labels (e.g., mapping the raw `med` prefix to *Healthcare*).

4. **Dimensionality Reduction & Visualization**
   * The 384-dimensional vectors were reduced to a 2D plane using **Principal Component Analysis (PCA)** to visually validate the structural integrity of the mathematical domains.

## Repository Structure

* **`/data/`**: Contains the raw Eurostat tables and NUTS 2021-2024 Excel files used as the foundation for the pipeline.
* **`/src/`**: Modular Python architecture containing data loaders (`loadData.py`, `extractData.py`), the classification engine (`classify.py`), the clustering algorithms (`cluster_measures.py`, `name_clusters.py`), and visualization logic (`clusterVis.py`).
* **`/outputs/`**: Contains the final categorized CSV files, the generated taxonomies, and the 2D PCA scatter plot (`cluster_plot.png`).
* **`/prompts/`**: Contains the exact, engineered text prompts fed to the Granite 3B model (`promptClassification`, `promptNameClsters`).

## Installation & Requirements

To run this pipeline locally, you must have [Ollama](https://ollama.com/) installed and running the `ibm/granite4.1:3b` model.

Install the required Python dependencies:
```bash
pip install pandas scikit-learn sentence-transformers matplotlib seaborn openpyxl
