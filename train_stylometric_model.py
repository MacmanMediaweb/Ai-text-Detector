import pandas as pd
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

def train_model(data_path, model_output_path):
    """
    Train and evaluate a Logistic Regression model using stylometric features.
    """
    # 1. Load the dataset
    print(f"--- Step 1: Loading Stylometric Features from {data_path} ---")
    if not os.path.exists(data_path):
        print(f"Error: {data_path} not found.")
        return
    
    df = pd.read_csv(data_path)

    # 2. Separate features (X) and label (y)
    # The label column is named 'label'
    X = df.drop('label', axis=1)
    y = df['label']

    # 3. Split dataset into 80% training and 20% testing
    print("\n--- Step 3: Splitting Dataset (80% Train, 20% Test) ---")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
    print(f"Training samples: {len(X_train)}")
    print(f"Testing samples: {len(X_test)}")

    # 4. Train a Logistic Regression model
    print("\n--- Step 4: Training Logistic Regression Model ---")
    # Using max_iter=1000 to ensure convergence
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train, y_train)

    # 5. Evaluate the model
    print("\n--- Step 5: Evaluating Model ---")
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    # 6. Display evaluation results
    print("\nEvaluation Metrics:")
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-score:  {f1:.4f}")

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Human (0)', 'AI (1)']))

    # 7. Save the trained model
    print(f"\n--- Step 7: Saving Model to {model_output_path} ---")
    # Ensure model directory exists
    os.makedirs(os.path.dirname(model_output_path), exist_ok=True)
    joblib.dump(model, model_output_path)
    print("Model saved successfully.")

if __name__ == "__main__":
    data_file = os.path.join('data', 'stylometric_features.csv')
    model_file = os.path.join('model', 'stylometric_model.pkl')
    train_model(data_file, model_file)
