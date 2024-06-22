from transformers.tokenization_utils import BatchEncoding, PreTrainedTokenizer
from parameters import MAX_SEQ_LEN

def tokenize(tokenizer: PreTrainedTokenizer, batch: list[str]) -> BatchEncoding:
    return tokenizer(batch, padding=True, truncation=True, return_tensors="pt", max_length=MAX_SEQ_LEN)