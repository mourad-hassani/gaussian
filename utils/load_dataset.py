import csv
from pathlib import Path
import pandas as pd
from parameters import INPUT_FILE_PATH

def load_dataset(file_path: Path = INPUT_FILE_PATH, start_index: int = 0, end_index: int = 100000) -> list[list[str]]:
    first_sentences: list[str] = []
    second_sentences: list[str] = []
    scores: list[float] = []
    with file_path.open("r") as f:
        reader = csv.reader(f)
        idx: int = 0
        for row in reader:
            if start_index <= idx <= end_index:
                first_sentences.append(row[1])
                second_sentences.append(row[2])
                scores.append(row[3])
            idx += 1
    return [first_sentences, second_sentences, scores]