import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

class DataloaderProvider:
    
    def __init__(self, dataset, batch_size, tokenizer):
        self.dataset = dataset
        self.tokenizer = tokenizer
        vocab = tokenizer.get_vocab()
        
        
        self.sos = vocab["[START]"]
        self.end = vocab["[END]"]
        self.unk = vocab["[UNK]"]
        self.pad = vocab["[MASK]"]
        
        processed_dataset = self.process(dataset)
        
        self.dataloader = DataLoader(
            processed_dataset, 
            batch_size=batch_size,
            collate_fn=self.pad_entry,
            shuffle=True, 
        )
        
    def process(self, dataset: Dataset):
        return dataset.map(self.tokenize)
    
    def tokenize(self, example):
        source = self.tokenizer.encode(example["translation"]["en"]).ids
        target = self.tokenizer.encode(example["translation"]["de"]).ids
        return {"src": source, "trg": target}
    
    def pad_entry(self, batch):
        max_eng_length = max(len(ex["src"]) for ex in batch)
        max_ger_length = max(len(ex["trg"]) for ex in batch)
        
        sources = [[self.sos] + ex["src"] + [self.end] + [self.pad] * (max_eng_length - len(ex["src"])) for ex in batch]
        targets = [[self.sos] + ex["trg"] + [self.end] + [self.pad] * (max_ger_length - len(ex["trg"])) for ex in batch]
        
        batch = {
            "sources": torch.tensor(sources, dtype=torch.long), 
            "targets": torch.tensor(targets, dtype=torch.long)
        }
        return batch
