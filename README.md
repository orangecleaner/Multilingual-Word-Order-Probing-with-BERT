# Multilingual-Word-Order-Probing-with-BERT

This project explores how multilingual BERT encodes **word order typology** (e.g., SOV, SVO, VSO) across languages.  
It uses probing classifiers and interpretability methods to analyze which BERT layers capture syntactic word order information.

## 📂 Project Structure

- **extract_embeddings.py**  
  Extracts sentence embeddings from all layers of `bert-base-multilingual-cased` for a given dataset (`word_order.json`).  
  The embeddings (per-layer [CLS] vectors) and labels are stored in `sentence_embeddings.npz`.

- **train_prob.py**  
  Trains an **MLP probe** on the extracted embeddings to classify word order (SOV/SVO/VSO).  
  - Runs experiments on multiple BERT layers.  
  - Logs accuracy and saves the best model (`mlp_probe_layer8.pt`).  
  - Outputs accuracy results in `acc_list.txt`.

- **predict_topology.py**  
  Loads the trained probe and applies **Integrated Gradients (Captum)** for interpretability.  
  - Visualizes token-level attributions for predicting word order.  
  - Supports multilingual text with font settings (Chinese, Japanese, Arabic, English).  
  - Saves visualizations as PNG files.

## ⚙️ Requirements

- Python 3.8+
- PyTorch
- Hugging Face Transformers
- scikit-learn
- Matplotlib
- Captum
- tqdm
- NumPy
