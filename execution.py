from gauss_model import GaussModel
import pandas as pd
from torch.utils.data import DataLoader
from utils.collate_fn import collate_fn
from parameters import BATCH_SIZE, SHUFFLE, NUM_WORKERS, DROP_lAST, MAX_SEQ_LEN, DTYPE, DEVICE, MODEL_NAME, INPUT_FILE_PATH, OUTPUT_DIRECTORY_PATH
from utils.create_optimizer import create_optimizer
from transformers.tokenization_utils import BatchEncoding, PreTrainedTokenizer
from transformers import AutoTokenizer
from utils.tokenize import tokenize
from gauss_model import GaussOutput
from typing import Callable
import torch
from utils.log_info import log_info
from tqdm import tqdm
from utils.similarity import asymmetrical_kl_sim

class Execution():
    def __init__(self):
        self.model: GaussModel = GaussModel(MODEL_NAME, True).eval().to(DEVICE)
        self.tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, model_max_length = MAX_SEQ_LEN, use_fast = False)

        self.train_dataset = pd.read_csv(str(INPUT_FILE_PATH)).to_dict("records")
        self.train_dataloader = DataLoader(self.train_dataset, collate_fn=collate_fn, batch_size=BATCH_SIZE, shuffle=SHUFFLE, num_workers=NUM_WORKERS, pin_memory=True, drop_last=DROP_lAST)

        self.optimizer, self.lr_scheduler = create_optimizer(model=self.model, train_steps_per_epoch=len(self.train_dataloader))

    def tokenize(self, batch: list[str]) -> BatchEncoding:
        return self.tokenizer(batch, padding=True, truncation=True, return_tensors="pt", max_length=MAX_SEQ_LEN)
    
    @torch.inference_mode()
    def encode_fn(self, sentences: list[str], **_) -> GaussOutput:
        self.model.eval()
        
        def my_collate_fn(batch):
            return self.tokenize(batch)

        data_loader = DataLoader(sentences, collate_fn=my_collate_fn, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True, drop_last=False)

        output: list[GaussOutput] = []
        for batch in data_loader:
            with torch.cuda.amp.autocast(dtype=DTYPE):
                out = self.model.forward(**batch.to(DEVICE))
            output.append(out)

        output = GaussOutput(
            mu=torch.cat([out.mu for out in output], dim=0),
            std=torch.cat([out.std for out in output], dim=0),
        )

        return output

    def log(self, metrics: dict) -> None:
        log_info(metrics, OUTPUT_DIRECTORY_PATH / "log.csv")
        tqdm.write(
            f"epoch: {metrics['epoch']} \t"
            f"step: {metrics['step']} \t"
            f"loss: {metrics['loss']:2.6f}       \t"
            f"dev-auc: {metrics['dev-auc']:.4f}"
        )
    
    def clone_state_dict(self) -> dict:
        return {k: v.detach().clone().cpu() for k, v in self.model.state_dict().items()}
    
    def sim_fn(self, sent0: list[str], sent1: list[str]) -> list[float]:
        sent0: GaussOutput = self.encode_fn(sent0)
        sent1: GaussOutput = self.encode_fn(sent1)
        return asymmetrical_kl_sim(sent0.mu, sent0.std, sent1.mu, sent1.std).tolist()