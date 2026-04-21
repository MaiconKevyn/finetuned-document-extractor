import json
import random

def split_dataset(input_file, train_file, val_file, split_ratio=0.9):
    with open(input_file, 'r') as f:
        lines = f.readlines()
    
    random.shuffle(lines)
    split_point = int(len(lines) * split_ratio)
    
    train_data = lines[:split_point]
    val_data = lines[split_point:]
    
    with open(train_file, 'w') as f:
        f.writelines(train_data)
        
    with open(val_file, 'w') as f:
        f.writelines(val_data)
        
    print(f"Split complete: {len(train_data)} training samples, {len(val_data)} validation samples.")

if __name__ == "__main__":
    split_dataset("data/train_dataset.jsonl", "data/train.jsonl", "data/val.jsonl")
