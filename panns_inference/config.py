import csv
import importlib.resources
from typing import List, Dict

# Constants
SAMPLE_RATE = 32000
CLASSES_NUM = None  # Will be set after loading labels

def load_label_data() -> tuple[List[str], List[str]]:
    """
    Load and parse the class labels CSV file from the package data.
    
    Returns:
        tuple: (list of label IDs, list of label names)
    """
    try:
        # Get the CSV file from package data
        with importlib.resources.files('panns_inference.data').joinpath('class_labels_indices.csv').open('r') as f:
            reader = csv.reader(f, delimiter=',')
            next(reader)  # Skip header row
            rows = list(reader)
        
        # Each label has a unique id such as "/m/068hy"
        ids = [row[1] for row in rows]
        labels = [row[2] for row in rows]
        
        return ids, labels
    except Exception as e:
        raise RuntimeError(f"Failed to load label data: {str(e)}")

# Load label data
ids, labels = load_label_data()

# Set the number of classes
CLASSES_NUM = len(labels)

# Mapping dictionaries
LB_TO_IX: Dict[str, int] = {label: i for i, label in enumerate(labels)}
IX_TO_LB: Dict[int, str] = {i: label for i, label in enumerate(labels)}
ID_TO_IX: Dict[str, int] = {id: i for i, id in enumerate(ids)}
IX_TO_ID: Dict[int, str] = {i: id for i, id in enumerate(ids)}

# Export these variables
__all__ = [
    'SAMPLE_RATE',
    'CLASSES_NUM',
    'labels',  # Add this for backward compatibility
    'classes_num',  # Add this for backward compatibility
    'LB_TO_IX',
    'IX_TO_LB',
    'ID_TO_IX',
    'IX_TO_ID',
]

# For backward compatibility
classes_num = CLASSES_NUM  # This line is important!