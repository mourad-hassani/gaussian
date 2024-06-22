import csv
from pathlib import Path
import pandas as pd

def load_dataset(file_path: Path = Path("./data/dataset.csv"), start_index: int = 0, end_index: int = 100000) -> list[list[str]]:
    first_sentences: list[str] = []
    second_sentences: list[str] = []
    scores: list[float] = []
    with file_path.open("r") as f:
        reader = csv.reader(f)
        idx: int = 0
        for row in reader:
            if start_index <= idx <= end_index:
                first_sentences.append(row[0])
                second_sentences.append(row[1])
                scores.append(row[2])
    return [first_sentences, second_sentences, scores]