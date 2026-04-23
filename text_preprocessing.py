import pandas as pd
import re
import os

def clean_text(text):
    """
    Preprocess text:
    - Convert to lowercase
    - Remove special characters (punctuation, symbols)
    - Remove extra whitespace
    """
    if not isinstance(text, str):
        return ""
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove special characters (keep only letters, numbers and basic spaces)
    # This keeps words intact while removing symbols/punctuation
    text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)
    
    # Remove extra whitespace (replace multiple spaces with a single space)
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def preprocess_dataset(input_path, output_path):
    """
    Load the dataset, apply cleaning, and save the result.
    """
    print(f"--- Loading dataset from {input_path} ---")
    if not os.path.exists(input_path):
        print(f"Error: {input_path} not found.")
        return
    
    df = pd.read_csv(input_path)
    
    # Show first 5 rows before preprocessing
    print("\n--- First 5 rows BEFORE preprocessing ---")
    print(df['text'].head())
    
    # Apply preprocessing
    print("\n--- Applying preprocessing (lowercase, symbols removal, whitespace) ---")
    df['text'] = df['text'].apply(clean_text)
    
    # Show first 5 rows after preprocessing
    print("\n--- First 5 rows AFTER preprocessing ---")
    print(df['text'].head())
    
    # Save the processed dataset
    print(f"\n--- Saving processed dataset to {output_path} ---")
    df.to_csv(output_path, index=False)
    print("Done!")

if __name__ == "__main__":
    # Ensure paths are correct for current directory
    input_file = "data/final_dataset.csv"
    output_file = "data/processed_dataset.csv"
    preprocess_dataset(input_file, output_file)
