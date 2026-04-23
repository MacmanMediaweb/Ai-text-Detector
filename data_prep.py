import pandas as pd
import os

def prepare_dataset(file_path, output_path):
    """
    Load, clean, standardize, balance, and save the dataset for AI vs Human text classification.
    """
    # 1. Load the dataset
    print(f"--- Step 1: Loading Dataset from {file_path} ---")
    if not os.path.exists(file_path):
        print(f"Error: File {file_path} not found.")
        return
    
    df = pd.read_csv(file_path)

    # 2. Display initial info
    print("\n--- Step 2: Initial Dataset Info ---")
    print("First 5 rows:")
    print(df.head())
    print("\nColumn names:", df.columns.tolist())
    print("Dataset shape:", df.shape)

    # 3. Identify text and label columns (Heuristically)
    print("\n--- Step 3: Identifying Columns ---")
    text_col = None
    label_col = None

    # Common names for text and label columns
    text_candidates = ['text', 'content', 'body', 'document', 'article', 'sentence']
    label_candidates = ['label', 'class', 'target', 'generated', 'category', 'ai']

    # Try to find by name first (case-insensitive)
    for col in df.columns:
        col_lower = col.lower()
        if not text_col and any(cand in col_lower for cand in text_candidates):
            text_col = col
        if not label_col and any(cand in col_lower for cand in label_candidates):
            label_col = col

    # Heuristics if not found by name
    if not text_col:
        # Longest average string length could be the text column
        text_col = df.select_dtypes(include=['object']).apply(lambda x: x.str.len().mean()).idxmax()
    
    if not label_col:
        # Column with fewest unique values (often the label)
        label_col = df.nunique().idxmin()

    print(f"Identified text column: '{text_col}'")
    print(f"Identified label column: '{label_col}'")

    # 4. Clean the dataset
    print("\n--- Step 4: Cleaning Dataset ---")
    initial_len = len(df)
    df = df.dropna(subset=[text_col, label_col])
    df = df.drop_duplicates(subset=[text_col])
    print(f"Removed {initial_len - len(df)} rows (missing values or duplicates).")

    # 5. Standardize column names
    print("\n--- Step 5: Standardizing Column Names ---")
    df = df[[text_col, label_col]].rename(columns={text_col: 'text', label_col: 'label'})
    print("Columns renamed to 'text' and 'label'.")

    # 6. Convert labels to numeric (Human = 0, AI = 1)
    print("\n--- Step 6: Converting Labels to Numeric ---")
    def convert_label(val):
        val_str = str(val).lower()
        if any(h in val_str for h in ['human', 'real', 'student', '0']):
            return 0
        if any(a in val_str for a in ['ai', 'generated', 'fake', 'machine', '1']):
            return 1
        return val # fallback if already numeric or unknown

    df['label'] = df['label'].apply(convert_label)
    
    # Ensure they are integers
    df['label'] = pd.to_numeric(df['label'], errors='coerce')
    df = df.dropna(subset=['label'])
    df['label'] = df['label'].astype(int)
    print("Labels converted: Human -> 0, AI -> 1")

    # 7. Balance the dataset
    print("\n--- Step 7: Balancing Dataset ---")
    class_counts = df['label'].value_counts()
    print("Class distribution before balancing:")
    print(class_counts)

    if len(class_counts) > 1:
        min_size = class_counts.min()
        df_balanced = df.groupby('label').apply(lambda x: x.sample(n=min_size, random_state=42)).reset_index(drop=True)
        df = df_balanced
        print(f"Balanced to {min_size} samples per class.")
    else:
        print("Warning: Only one class found. Balancing skipped.")

    # 8. Shuffle the dataset
    print("\n--- Step 8: Shuffling Dataset ---")
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    # 9. Save cleaned dataset
    print(f"\n--- Step 9: Saving Final Dataset to {output_path} ---")
    df.to_csv(output_path, index=False)
    print("File saved successfully.")

    # 10. Print final stats
    print("\n--- Step 10: Final Dataset Stats ---")
    print("Final shape:", df.shape)
    print("Label distribution:")
    print(df['label'].value_counts())

if __name__ == "__main__":
    input_csv = os.path.join('data', 'dataset.csv')
    output_csv = os.path.join('data', 'final_dataset.csv')
    prepare_dataset(input_csv, output_csv)
