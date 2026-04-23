import pandas as pd
import numpy as np
import torch
import os
import joblib
from tqdm import tqdm
from transformers import DistilBertTokenizer, DistilBertModel
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

def train_transformer_classifier(data_path, model_output_path, sample_size=2000):
    """
    Extract [CLS] embeddings from DistilBERT on CPU and train a Logistic Regression classifier.
    """
    # 1. Load dataset
    print(f"--- Step 1: Loading Processed Dataset from {data_path} ---")
    if not os.path.exists(data_path):
        print(f"Error: {data_path} not found.")
        return
    
    df = pd.read_csv(data_path)
    
    # 2. Sample data for CPU-friendly extraction
    print(f"Total samples: {len(df)}. Sampling {sample_size} for CPU-efficient processing.")
    df = df.sample(sample_size, random_state=42).reset_index(drop=True)
    
    texts = df['text'].astype(str).tolist()
    labels = df['label'].values

    # 3. Load DistilBERT (CPU only)
    print("\n--- Step 3: Loading DistilBERT (CPU Mode) ---")
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    model = DistilBertModel.from_pretrained('distilbert-base-uncased')
    
    # Force CPU
    device = torch.device('cpu')
    model.to(device)
    model.eval()

    # 4. Extract [CLS] Embeddings
    print("\n--- Step 4: Extracting [CLS] Token Embeddings ---")
    all_embeddings = []
    batch_size = 16
    max_length = 128
    
    with torch.no_grad():
        for i in tqdm(range(0, len(texts), batch_size), desc="Extracting"):
            batch_texts = texts[i:i + batch_size]
            
            # Tokenize and encode
            inputs = tokenizer(
                batch_texts, 
                padding=True, 
                truncation=True, 
                max_length=max_length, 
                return_tensors='pt'
            ).to(device)
            
            # Forward pass
            outputs = model(**inputs)
            
            # Get [CLS] embedding (first token)
            cls_emb = outputs.last_hidden_state[:, 0, :].cpu().numpy()
            all_embeddings.append(cls_emb)

    X = np.vstack(all_embeddings)
    y = labels

    # 5. Train/Test Split (80% / 20%)
    print("\n--- Step 5: Splitting Data (80% Train, 20% Test) ---")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

    # 6. Train Lightweight Classifier (Logistic Regression)
    print("\n--- Step 6: Training Logistic Regression Classifier ---")
    classifier = LogisticRegression(max_iter=1000, random_state=42)
    classifier.fit(X_train, y_train)

    # 7. Evaluate
    print("\n--- Step 7: Evaluating Classifier ---")
    y_pred = classifier.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    print("\nEvaluation Results (Transformer + LR):")
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-score:  {f1:.4f}")

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Human (0)', 'AI (1)']))

    # 8. Save Classifier
    print(f"\n--- Step 8: Saving Classifier to {model_output_path} ---")
    os.makedirs(os.path.dirname(model_output_path), exist_ok=True)
    joblib.dump(classifier, model_output_path)
    print("Model saved successfully.")

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))
    input_file = os.path.join(base_dir, 'data', 'processed_dataset.csv')
    output_model = os.path.join(base_dir, 'model', 'transformer_classifier.pkl')
    
    # Process a representative subset for comparison
    train_transformer_classifier(input_file, output_model, sample_size=500)
