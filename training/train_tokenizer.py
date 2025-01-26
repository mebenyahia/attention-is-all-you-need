import json
import os
import config
from datasets import Dataset
from tokenizers import Tokenizer, trainers, models, pre_tokenizers, normalizers

allowed_sequence_length = config.ALLOWED_SEQ_LENGTH

def load_from_json(filename):
    with open(filename, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return Dataset.from_list(data)

langs = config.LANGS

print("Loading data splits from saved...")

# load train, validation and test splits
train_dataset = load_from_json(f'data/{langs}-train_data.json')
validation_dataset = load_from_json(f'data/{langs}-validation_data.json')
test_dataset = load_from_json(f'data/{langs}-test_data.json')

print("Preparing data for tokenizer training...")
os.makedirs('tokenizer', exist_ok=True)
with open("tokenizer/common_data.txt", "w", encoding="utf-8") as common_file:
    for split in [train_dataset, validation_dataset, test_dataset]:
        for example in split:
            common_file.write(example['translation'][config.LANG_1] + "\n")
            common_file.write(example['translation'][config.LANG_2] + "\n")
            
tokenizer = Tokenizer(models.BPE(unk_token='[UNK]'))

tokenizer.normalizer = normalizers.Sequence([
     normalizers.Lowercase()
])
tokenizer.pre_tokenizer = pre_tokenizers.Sequence([
     pre_tokenizers.Whitespace(),
])
tokenizer.enable_truncation(max_length=allowed_sequence_length)
trainer = trainers.BpeTrainer(vocab_size=config.VOCAB_SIZE, special_tokens=["[UNK]", "[START]", "[END]", "[PAD]"], show_progress=True)

print("Training tokenizer...")
os.environ["TOKENIZERS_PARALLELISM"] = "true"
tokenizer.train(["tokenizer/common_data.txt"], trainer=trainer)

print("Saving tokenizer...")
tokenizer.save('tokenizer/bpe.json')

print("Finished.")