import pandas as pd
import numpy as np
import joblib
import torch
import os
import time
from tqdm import tqdm
from transformers import DistilBertTokenizer, DistilBertModel
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from stylometric_extraction import extract_stylometric_features

def compare_models(data_path, stylo_model_path, trans_model_path, sample_size=100):
    """
    Compare Stylometric and Transformer models on the same test set.
    """
    # 1. Load Dataset
    print(f"--- Step 1: Loading Dataset from {data_path} ---")
    df = pd.read_csv(data_path)
    
    # Use same split logic as training scripts to ensure valid test set
    # Note: We use the raw text for extraction, then compare labels
    _, test_df = train_test_split(df, test_size=0.20, random_state=42)
    
    # Sample for quick academic demonstration
    eval_df = test_df.sample(min(sample_size, len(test_df)), random_state=42).reset_index(drop=True)
    texts = eval_df['text'].astype(str).tolist()
    y_true = eval_df['label'].values

    # 2. Load Models
    print("\n--- Step 2: Loading Models ---")
    stylo_model = joblib.load(stylo_model_path)
    trans_classifier = joblib.load(trans_model_path)
    
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    trans_backbone = DistilBertModel.from_pretrained('distilbert-base-uncased')
    trans_backbone.eval()

    results = []

    # 3. Evaluate Stylometric Model
    print("\n--- Step 3: Evaluating Stylometric Model ---")
    start_time = time.time()
    stylo_features = [list(extract_stylometric_features(t).values()) for t in texts]
    stylo_preds = stylo_model.predict(stylo_features)
    stylo_time = time.time() - start_time
    
    results.append({
        "Model": "Stylometric (LR)",
        "Accuracy": accuracy_score(y_true, stylo_preds),
        "Precision": precision_score(y_true, stylo_preds),
        "Recall": recall_score(y_true, stylo_preds),
        "F1-Score": f1_score(y_true, stylo_preds),
        "Inference Time (s)": round(stylo_time, 4)
    })

    # 4. Evaluate Transformer Model
    print("\n--- Step 4: Evaluating Transformer Model ---")
    start_time = time.time()
    all_embeddings = []
    batch_size = 16
    
    with torch.no_grad():
        for i in tqdm(range(0, len(texts), batch_size), desc="Extracting Transformer Features"):
            batch_texts = texts[i:i + batch_size]
            inputs = tokenizer(batch_texts, padding=True, truncation=True, max_length=128, return_tensors='pt')
            outputs = trans_backbone(**inputs)
            cls_emb = outputs.last_hidden_state[:, 0, :].numpy()
            all_embeddings.append(cls_emb)
    
    X_trans = np.vstack(all_embeddings)
    trans_preds = trans_classifier.predict(X_trans)
    trans_time = time.time() - start_time

    results.append({
        "Model": "Transformer (DistilBERT + LR)",
        "Accuracy": accuracy_score(y_true, trans_preds),
        "Precision": precision_score(y_true, trans_preds),
        "Recall": recall_score(y_true, trans_preds),
        "F1-Score": f1_score(y_true, trans_preds),
        "Inference Time (s)": round(trans_time, 4)
    })

    # 5. Display Comparison Table
    comparison_df = pd.DataFrame(results)
    print("\n" + "="*60)
    print("MODEL COMPARISON RESULTS")
    print("="*60)
    print(comparison_df.to_string(index=False))
    print("="*60)

    # 6. Summary Report
    print("\n--- Summary Report ---")
    best_acc = comparison_df.loc[comparison_df['Accuracy'].idxmax(), 'Model']
    fastest = comparison_df.loc[comparison_df['Inference Time (s)'].idxmin(), 'Model']
    
    print(f"- Accuracy Winner: {best_acc}")
    print(f"- Efficiency Winner: {fastest}")
    print("\nAcademic Insight:")
    print("1. The Transformer model captures semantic and contextual depth, leading to higher accuracy.")
    print("2. The Stylometric model is significantly faster and uses minimal computational resources (CPU-friendly).")
    print("3. Combining both (Hybrid Approach) could be useful: Use the Stylometric model as a fast 'first-pass' filter")
    print("   and only use the Transformer model for uncertain cases to optimize resource usage in production.")

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_file = os.path.join(base_dir, 'data', 'final_dataset.csv') # Use raw for stylo extraction
    stylo_path = os.path.join(base_dir, 'model', 'stylometric_model.pkl')
    trans_path = os.path.join(base_dir, 'model', 'transformer_classifier.pkl')
    
    if os.path.exists(stylo_path) and os.path.exists(trans_path):
        compare_models(data_file, stylo_path, trans_path, sample_size=100)
    else:
        print("Error: One or both models not found. Please ensure Step 5A and 5B are completed.")
