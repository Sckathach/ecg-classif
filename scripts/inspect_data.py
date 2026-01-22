import pickle
import sys

try:
    with open("data/processed/features.pkl", "rb") as f:
        data = pickle.load(f)
    
    print(f"Top level keys: {list(data.keys())}")
    
    if len(data) > 0:
        first_key = list(data.keys())[0]
        if isinstance(data[first_key], dict):
            print(f"Second level keys (example for {first_key}): {list(data[first_key].keys())}")
            
            first_subkey = list(data[first_key].keys())[0]
            if isinstance(data[first_key][first_subkey], dict):
                print(f"Third level keys (example for {first_key}/{first_subkey}): {list(data[first_key][first_subkey].keys())}")
                print(f"Shape of X for {first_key}/{first_subkey}: {data[first_key][first_subkey]['X'].shape}")
                print(f"Shape of y for {first_key}/{first_subkey}: {data[first_key][first_subkey]['y'].shape}")
                
except Exception as e:
    print(f"Error reading pickle: {e}")
