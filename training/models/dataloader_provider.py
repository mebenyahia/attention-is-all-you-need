import torch
from torch.utils.data import DataLoader

class DataloaderProvider:
    
    def __init__(self, dataset, batch_size, tokenizer):
        self.dataset = dataset
        self.tokenizer = tokenizer
        vocab = tokenizer.get_vocab()
        
        self.sos = vocab["[START]"]
        self.end = vocab["[END]"]
        self.unk = vocab["[UNK]"]
        self.mask = vocab["[MASK]"]
        
        # TODO: tokenize the dataset outside the collate_fn function to improve training speed
        # self.tokenized = []
        
        def collate_fn(batch):
            tokenized_batch = []
            for example in batch:
                # Tokenize and encode sentences
                eng = self.tokenizer.encode(example["translation"]["en"]).ids
                ger = self.tokenizer.encode(example["translation"]["de"]).ids
            
                tokenized_batch.append((eng, ger))
        
            # Find the maximum sequence length in the batch
            max_eng_length = max(len(eng) for (eng, _) in tokenized_batch)
            max_ger_length = max(len(ger) for (_, ger) in tokenized_batch)

            # Pad all sequences to the maximum length (we pad with the end of sentence)
            sources = [[self.sos] + eng + [self.end] * (max_eng_length - len(eng)) + [self.end] for eng, _ in tokenized_batch]
            targets = [[self.sos] + ger + [self.end] * (max_ger_length - len(ger)) + [self.end] for _, ger in tokenized_batch]
        
            batch = {
               "sources": torch.tensor(sources, dtype=torch.long), 
               "targets": torch.tensor(targets, dtype=torch.long)
            }
            return batch
        
        self.dataloader = DataLoader(
            dataset, 
            batch_size=batch_size, 
            shuffle=True, 
            collate_fn=collate_fn
        )
  