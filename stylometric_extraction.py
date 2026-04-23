import pandas as pd
import re
import os
import string

def extract_stylometric_features(text):
    """
    Extract stylometric features from text:
    - Average sentence length
    - Total word count
    - Punctuation count
    - Average word length
    """
    if not isinstance(text, str) or not text.strip():
        return {
            "avg_sentence_len": 0,
            "total_word_count": 0,
            "punctuation_count": 0,
            "avg_word_len": 0
        }
    
    # 1. Total word count
    words = text.split()
    total_word_count = len(words)
    
    # 2. Punctuation count
    # Count all characters that are in the standard punctuation list
    punctuation_count = sum(1 for char in text if char in string.punctuation)
    
    # 3. Average word length
    if total_word_count > 0:
        avg_word_len = sum(len(word) for word in words) / total_word_count
    else:
        avg_word_len = 0
        
    # 4. Average sentence length
    # Split sentences by . ! ?
    sentences = re.split(r'[.!?]+', text)
    # Filter out empty strings from splitting
    sentences = [s.strip() for s in sentences if s.strip()]
    
    if len(sentences) > 0:
        avg_sentence_len = total_word_count / len(sentences)
    else:
        avg_sentence_len = total_word_count # assume 1 sentence if no punctuation
        
    return {
        "avg_sentence_len": round(avg_sentence_len, 2),
        "total_word_count": total_word_count,
        "punctuation_count": punctuation_count,
        "avg_word_len": round(avg_word_len, 2)
    }

def process_features(input_path, output_path):
    """
    Load dataset, apply feature extraction, and save features to a new file.
    """
    print(f"--- Loading dataset from {input_path} ---")
    if not os.path.exists(input_path):
        print(f"Error: {input_path} not found.")
        return
    
    df = pd.read_csv(input_path)
    
    print("\n--- Extracting Stylometric Features ---")
    # Apply the extraction function and expand results into separate columns
    features_df = df['text'].apply(extract_stylometric_features).apply(pd.Series)
    
    # Combine features with the label from the original dataset
    final_df = pd.concat([features_df, df['label']], axis=1)
    
    print("\n--- Feature Extraction Complete ---")
    print("First 5 rows of extracted features:")
    print(final_df.head())
    
    print(f"\nNew dataset shape: {final_df.shape}")
    
    # Save the features dataset
    print(f"\nSaving features to {output_path}...")
    final_df.to_csv(output_path, index=False)
    print("Done!")

if __name__ == "__main__":
    # Note: We use final_dataset.csv for stylometric extraction because processed_dataset.csv 
    # has punctuation removed, which would make punctuation count 0 and sentence length 1.
    input_file = "data/final_dataset.csv"
    output_file = "data/stylometric_features.csv"
    process_features(input_file, output_file)
