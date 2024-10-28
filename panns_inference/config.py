import csv
import os
from pathlib import Path
from typing import List, Dict
import sys

SAMPLE_RATE = 32000
CLASSES_NUM = None  


def load_label_data() -> tuple[List[str], List[str]]:
    """
    Load and parse the class labels CSV file from the package data.
    Windows-safe path handling via the Path library
    
    Returns:
        tuple: (list of label IDs, list of label names)
    """
    try:

        import panns_inference
        package_dir = Path(panns_inference.__file__).parent
        
        csv_path = package_dir.joinpath('data', 'class_labels_indices.csv') #join path
        
        csv_path = csv_path.absolute() # Important : otherwise windows path crashs
        
        if not csv_path.exists():
            raise FileNotFoundError(f"CSV file not found at {csv_path}")
            
        with csv_path.open('r', encoding='utf-8') as f:

            reader = csv.reader(f)
            next(reader) 
            rows = list(reader)
            
            if not rows:
                raise ValueError("CSV file is empty")
            
            ids = [row[1] for row in rows]
            labels = [row[2] for row in rows]
            
            if not ids or not labels:
                raise ValueError("No labels found in CSV file")
            
            return ids, labels
            
    except Exception as e:
        print(f"Failed to load labels: {str(e)}")
        print(f"Attempted path: {csv_path}")
        print(f"Current platform: {sys.platform}")
        print(f"Package directory: {package_dir}")
        raise RuntimeError("Failed to load label data")


try:
    ids, labels = load_label_data()
    CLASSES_NUM = len(labels)
    print(f"Successfully loaded {CLASSES_NUM} classes")

except Exception as e:

    print(f"Warning: Failed to load label data: {str(e)}")
    # Initialize with the expected number of classes based on the checkpoint
    print("Falling back to default number of classes (527)")
    CLASSES_NUM = 527
    ids = [f"id_{i}" for i in range(CLASSES_NUM)]
    labels = [f"label_{i}" for i in range(CLASSES_NUM)]

LB_TO_IX: Dict[str, int] = {label: i for i, label in enumerate(labels)}
IX_TO_LB: Dict[int, str] = {i: label for i, label in enumerate(labels)}
ID_TO_IX: Dict[str, int] = {id: i for i, id in enumerate(ids)}
IX_TO_ID: Dict[int, str] = {i: id for i, id in enumerate(ids)}


__all__ = [
    'SAMPLE_RATE',
    'CLASSES_NUM',
    'labels',  # For backward compatibility
    'classes_num',  # For backward compatibility
    'LB_TO_IX',
    'IX_TO_LB',
    'ID_TO_IX',
    'IX_TO_ID',
]

# For backward compatibility
classes_num = CLASSES_NUM