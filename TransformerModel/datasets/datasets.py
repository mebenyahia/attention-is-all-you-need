import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator

class WMT14Dataset(Dataset):
    def __init__(self, filepath, language_pair=('en', 'fr'), root='.data/wmt14'):
        super(WMT14Dataset, self).__init__()
        self.filepath = filepath
        self.language_pair = language_pair
        
        self.tokenizer_src = get_tokenizer('spacy', language='en_core_web_sm')
        self.tokenizer_trg = get_tokenizer('spacy', language='fr_core_news_sm')

        self.data = self.load_data()

        # Build vocabularies based on the tokenizer
        self.vocab_src = build_vocab_from_iterator(map(self.tokenizer_src, (src for src, trg in self.data)), specials=['<unk>', '<pad>', '<bos>', '<eos>'])
        self.vocab_trg = build_vocab_from_iterator(map(self.tokenizer_trg, (trg for src, trg in self.data)), specials=['<unk>', '<pad>', '<bos>', '<eos>'])

        self.vocab_src.set_default_index(self.vocab_src['<unk>'])
        self.vocab_trg.set_default_index(self.vocab_trg['<unk>'])

    def load_data(self):
        # Load CSV file
        df = pd.read_csv(self.filepath)
        src_lines = df[self.language_pair[0]].tolist()
        trg_lines = df[self.language_pair[1]].tolist()

        return list(zip(src_lines, trg_lines))

    def __getitem__(self, index):
        src_line, trg_line = self.data[index]
        src_tensor = torch.tensor([self.vocab_src[token] for token in self.tokenizer_src(src_line)], dtype=torch.long)
        trg_tensor = torch.tensor([self.vocab_trg[token] for token in self.tokenizer_trg(trg_line)], dtype=torch.long)
        return src_tensor, trg_tensor

    def __len__(self):
        return len(self.data)

def collate_fn(batch):
    src_batch, trg_batch = [], []
    for src_item, trg_item in batch:
        src_batch.append(torch.cat([torch.tensor([2]), src_item, torch.tensor([3])], dim=0))  # 2 is <bos>, 3 is <eos>
        trg_batch.append(torch.cat([torch.tensor([2]), trg_item, torch.tensor([3])], dim=0))  # 2 is <bos>, 3 is <eos>

    src_batch = torch.nn.utils.rnn.pad_sequence(src_batch, padding_value=1)  # 1 is <pad>
    trg_batch = torch.nn.utils.rnn.pad_sequence(trg_batch, padding_value=1)  # 1 is <pad>
    return src_batch, trg_batch
