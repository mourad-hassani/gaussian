import csv
from pathlib import Path
import pandas as pd
from parameters import INPUT_FILE_PATH

def load_dataset(file_path: Path = INPUT_FILE_PATH, split: str = "train") -> list[list[str]]:
    first_sentences: list[str] = []
    second_sentences: list[str] = []
    scores: list[float] = []
    with file_path.open("r") as f:
        reader = csv.reader(f)
        row_count: int = 0
        for row in reader:
            row_count += 1
            
        if split == "train":
            start_index = 0
            end_index = int(row_count * 0.8)
        elif split == "dev":
            start_index = int(row_count * 0.8) + 1
            end_index = int(row_count * 0.9)
        elif split == "test":
            start_index = int(row_count * 0.9) + 1
            end_index = int(row_count)
        idx: int = 0
        for row in reader:
            if start_index <= idx <= end_index:
                first_sentences.append(row[0])
                second_sentences.append(row[1])
                scores.append(row[2])
            idx += 1
    return [first_sentences, second_sentences, scores]