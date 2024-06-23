from pathlib import Path
from typing import Callable
from utils.load_dataset import load_dataset
from parameters import INPUT_FILE_PATH

import torch
from scipy.stats import spearmanr

class EvaluatorBase:
    def __init__(self, dataset_path: Path = INPUT_FILE_PATH, split: str = "train"):
        if split == "train":
            self.sentences1, self.sentences2, self.scores = load_dataset(dataset_path, split="train")
        elif split == "dev":
            self.sentences1, self.sentences2, self.scores = load_dataset(dataset_path, split="dev")
        assert len(self.sentences1) == len(self.sentences2) == len(self.scores)

    def __call__(self, sim_fn: Callable[[list[str], list[str]], list[float]]) -> float:
        similarities = sim_fn(self.sentences1, self.sentences2)
        spearman = float(spearmanr(self.scores, similarities)[0]) * 100
        return spearman
    
class TestEvaluatorBase:
    def __init__(self, dataset_path: Path = INPUT_FILE_PATH):
        self.sentences1, self.sentences2, self.scores = load_dataset(dataset_path, split="test")
        assert len(self.sentences1) == len(self.sentences2) == len(self.scores)

    def __call__(self, sim_fn: Callable[[list[str], list[str]], list[float]]) -> float:
        similarities = sim_fn(self.sentences1, self.sentences2)
        spearman = float(spearmanr(self.scores, similarities)[0]) * 100
        return {"sent0": self.sentences1, "sent1": self.sentences2, "similarity": similarities,"score": spearman}

class Evaluator:
    def __init__(self, sim_fn: Callable[[list[str], list[str]], list[float]]) -> None:
        self.sim_fn = sim_fn
        self.evaluator = EvaluatorBase()
        self.dev_evaluator = EvaluatorBase(split="dev")
        self.test_evaluator = TestEvaluatorBase()

    @torch.inference_mode()
    def eval(self) -> list[float]:
        return self.evaluator(self.sim_fn)

    @torch.inference_mode()
    def dev(self) -> float:
        return self.dev_evaluator(self.sim_fn)
    
    @torch.inference_mode()
    def test(self) -> dict:
        return self.test_evaluator(self.sim_fn)