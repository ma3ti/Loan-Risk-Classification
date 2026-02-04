import pandas as pd
import numpy as np
import sys
import os
import test  # Imports your test.py

def run_full_check():
    print("=== STARTING FULL SYSTEM CHECK ===")
    
    # 1. Load Dummy Data
    data_path = "data/train.csv"
    if not os.path.exists(data_path):
        print(f"ERROR: Data not found at {data_path}")
        return

    try:
        df_full = pd.read_csv(data_path)
        
        # Take a small sample (3 rows)
        # Using a fixed random_state ensures we get the same rows every time
        sample = df_full.sample(10, random_state=42)
        
        # Separate Features (X) and Target (y)
        # We assume the target column is named "grade"
        if "grade" in sample.columns:
            y_true = sample["grade"].values
            X_raw = sample.drop(columns=["grade"])
        else:
            y_true = ["N/A"] * len(sample)
            X_raw = sample
            
        print(f"Data loaded. Testing on {len(sample)} rows.")
        print(f"True Labels for this sample: {y_true}")
        
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    # 2. List of models to test
    methods = ['rf', 'knn', 'svm', 'ffnn', 'tabnet', 'tabtransformer']
    
    for m in methods:
        print(f"\n{'='*30}")
        print(f"TESTING METHOD: {m.upper()}")
        print(f"{'='*30}")
        
        try:
            # A. Preprocess
            print(f"1. Preprocessing ({m})...")
            X_proc = test.preprocess(X_raw, clfName=m)
            
            shape_info = X_proc.shape if hasattr(X_proc, "shape") else "Unknown"
            print(f"   Success. Output shape: {shape_info}")
            
            # B. Load Model
            print(f"2. Loading Model ({m})...")
            clf = test.load(clfName=m)
            
            # C. Predict
            print(f"3. Predicting ({m})...")
            preds = test.predict(X_proc, clf)
            
            # D. Compare
            print(f"   Predictions: {preds}")
            print(f"   True Labels: {y_true}")
            
            # Simple accuracy check for this tiny batch
            correct = np.sum(preds == y_true)
            print(f"   Match: {correct}/{len(preds)}")
            
        except FileNotFoundError as fnf:
            print(f"   SKIP: File missing - {fnf}")
        except Exception as e:
            print(f"   FAIL: Error testing {m}")
            print(f"   Details: {e}")

    print("\n=== CHECK COMPLETE ===")

if __name__ == "__main__":
    run_full_check()