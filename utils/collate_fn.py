from transformers.tokenization_utils import BatchEncoding

def collate_fn(self, data_list: list[dict]) -> BatchEncoding:
        return BatchEncoding(
            {
                "sent0": self.tokenize([d["sent0"] for d in data_list]),
                "sent1": self.tokenize([d["sent1"] for d in data_list]),
                "score": self.tokenize([d["score"] for d in data_list]),
            }
        )