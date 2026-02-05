import pandas as pd
import numpy as np
import sys
import os
import test  # Imports your updated test.py

def run_full_check():
    print("=== STARTING METRIC SYSTEM CHECK ===")
    
    # 1. Load Dummy Data
    data_path = "data/train.csv"
    if not os.path.exists(data_path):
        print(f"ERROR: Data not found at {data_path}")
        return

    try:
        df_full = pd.read_csv(data_path)
        sample = df_full.sample(10, random_state=42)
        print(f"Data loaded. Testing on {len(sample)} rows.")
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    # 2. List of models
    methods = ['lr', 'rf', 'knn', 'svm', 'ff', 'tb', 'tf']
    
    for m in methods:
        print(f"\n{'='*30}")
        print(f"TESTING METHOD: {m.upper()}")
        print(f"{'='*30}")
        
        try:
            # A. Preprocess (Returns X AND y now)
            print(f"Preprocessing ({m})...")
            # Note: We pass the full sample. preprocess handles splitting X and y.
            dataset_processed = test.preprocess(sample.copy(), clfName=m)
            
            # B. Load Model
            print(f"Loading Model ({m})...")
            clf = test.load(clfName=m)
            
            # C. Predict & Evaluate
            print(f"Predicting & Scoring ({m})...")
            results = test.predict(dataset_processed, clf)
            
            # D. Show Results
            print("RESULTS DICTIONARY:")
            print(f"   Accuracy:          {results['acc']:.4f}")
            print(f"   Balanced Accuracy: {results['bacc']:.4f}")
            print(f"   F1 Score (W):      {results['f1']:.4f}")
            print(f"   Sample Preds:      {results['predictions'][:3]}")
            
        except FileNotFoundError as fnf:
            print(f"   SKIP: File missing - {fnf}")
        except Exception as e:
            print(f"   FAIL: Error testing {m}")
            print(f"   Details: {e}")
            import traceback
            traceback.print_exc()

    print("\n=== CHECK COMPLETE ===")

if __name__ == "__main__":
    run_full_check()