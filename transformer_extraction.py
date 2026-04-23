import pandas as pd
import numpy as np
import torch
import os
from transformers import DistilBertTokenizer, DistilBertModel

def extract_features():
    # Use absolute path to ensure data directory is found
    data_dir = os.path.join(os.getcwd(), "data")
    input_path = os.path.join(data_dir, "processed_dataset.csv")
    features_path = os.path.join(data_dir, "transformer_features.npy")
    labels_path = os.path.join(data_dir, "transformer_labels.npy")

    print(f"--- Loading from: {input_path} ---")
    df = pd.read_csv(input_path).head(100)
    print(f"Processing {len(df)} samples...")
    
    tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
    model = DistilBertModel.from_pretrained("distilbert-base-uncased")
    model.eval()
    
    texts = df["text"].astype(str).tolist()
    labels = df["label"].values
    
    all_embs = []
    batch_size = 10
    
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            inputs = tokenizer(batch, padding=True, truncation=True, max_length=128, return_tensors="pt")
            outputs = model(**inputs)
            cls_emb = outputs.last_hidden_state[:, 0, :].cpu().numpy()
            all_embs.append(cls_emb)
            
    final_embs = np.vstack(all_embs)
    
    # Save using absolute paths
    np.save(features_path, final_embs)
    np.save(labels_path, labels)
    
    print(f"--- Saved successfully to {data_dir} ---")
    print(f"Features shape: {final_embs.shape}")

if __name__ == "__main__":
    extract_features()
