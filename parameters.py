from pathlib import Path
import torch

MODEL_NAME: str = "prajjwal1/bert-small"

BATCH_SIZE: int = 64
SHUFFLE: bool = True
NUM_WORKERS: int = 4
DROP_lAST: bool = True
LR: float = 3e-5
WEIGHT_DECAY: float = 1e-2
EPOCHS: int = 1
NUM_WARMUP_RATIO: float = 0.1
MAX_SEQ_LEN: int = 64
DEVICE: str = "cuda:0"
DTYPE: torch.dtype = torch.float16
SEED: int = 0
TEMPERATURE: float = 0.05
NUM_EVAL_STEPS: int = 1000

INPUT_FILE_PATH: str = Path("data/train_dataset.csv")
OUTPUT_DIRECTORY_PATH: Path = Path("output")