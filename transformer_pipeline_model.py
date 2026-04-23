import pandas as pd
from transformers import pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
import os
from tqdm import tqdm

def run_transformer_pipeline(data_path, sample_size=50):
    """
    Run AI text detection using a pretrained pipeline.
    Uses a sample of the test set for efficient evaluation in CPU mode.
    """
    # 1. Load the processed dataset
    print(f"--- Step 1: Loading Dataset from {data_path} ---")
    if not os.path.exists(data_path):
        print(f"Error: {data_path} not found.")
        return
    
    df = pd.read_csv(data_path)
    df['label'] = df['label'].astype(int)
    
    # 3. Split dataset
    print("\n--- Step 3: Splitting Dataset ---")
    _, test_df = train_test_split(df, test_size=0.20, random_state=42)
    
    # Sample for CPU evaluation
    print(f"Sampling {sample_size} for quick evaluation on CPU.")
    eval_df = test_df.sample(sample_size, random_state=42).reset_index(drop=True)
    
    X_test = eval_df['text'].astype(str).tolist()
    y_test = eval_df['label'].tolist()

    # 4. Initialize Pipeline
    print("\n--- Step 4: Initializing Pipeline (DistilBERT Zero-Shot) ---")
    # Using a model that supports zero-shot to distinguish between categories without training
    classifier = pipeline(
        "zero-shot-classification", 
        model="typeform/distilbert-base-uncased-mnli", 
        device=-1
    )
    
    candidate_labels = ["human-written", "AI-generated"]

    # 5. Make Predictions
    print("\n--- Step 5: Making Predictions ---")
    y_pred = []
    
    for text in tqdm(X_test, desc="Processing"):
        # Truncate to 512 for DistilBERT
        result = classifier(text[:512], candidate_labels=candidate_labels)
        top_label = result['labels'][0]
        y_pred.append(1 if top_label == "AI-generated" else 0)

    # 6. Evaluate
    print("\n--- Step 6: Evaluating Results ---")
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Human (0)', 'AI (1)']))

if __name__ == "__main__":
    input_file = "data/processed_dataset.csv"
    run_transformer_pipeline(input_file, sample_size=20)
