import os
import urllib.request
import numpy as np
import csv
from pathlib import Path

sample_rate = 32000
labels_csv_path = Path.home() / 'panns_data' / 'class_labels_indices.csv'

if not labels_csv_path.exists():
    labels_csv_path.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        print(f"Downloading labels to {labels_csv_path}")
        url = 'http://storage.googleapis.com/us_audioset/youtube_corpus/v1/csv/class_labels_indices.csv'
        with urllib.request.urlopen(url) as ifh:
            labels_csv_path.write_bytes(ifh.read())
    except Exception as e:
        print(f"Error downloading labels: {e}")
        raise

# Load label
try:
    with labels_csv_path.open('r') as f:
        reader = csv.reader(f, delimiter=',')
        lines = list(reader)
except Exception as e:
    print(f"Error reading labels from {labels_csv_path}: {e}")
    raise


labels = []
ids = []    # Each label has a unique id such as "/m/068hy"
for i1 in range(1, len(lines)):
    id = lines[i1][1]
    label = lines[i1][2]
    ids.append(id)
    labels.append(label)

classes_num = len(labels)

lb_to_ix = {label : i for i, label in enumerate(labels)}
ix_to_lb = {i : label for i, label in enumerate(labels)}

id_to_ix = {id : i for i, id in enumerate(ids)}
ix_to_id = {i : id for i, id in enumerate(ids)}