import json

import torch
from accelerate import Accelerator
from datasets import Dataset
from tokenizers import Tokenizer
from models import Transformer
import config
from models import DataloaderProvider
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

# Initialize the accelerator
output_dir = "saved_model"
accelerator = Accelerator()
tokenizer = Tokenizer.from_file('tokenizer/bpe.json')
vocab = tokenizer.get_vocab()
sos = vocab["[START]"]
end = vocab["[END]"]
pad = vocab["[PAD]"]

# Load the trained model
model = Transformer(config.VOCAB_SIZE, config.D_MODEL, config.D_FF, config.N_HEADS, config.N_LAYERS, config.ALLOWED_SEQ_LENGTH, pad_token=pad)
model = accelerator.prepare(model)
accelerator.load_state(output_dir)

def load_from_json(filename):
    with open(filename, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return Dataset.from_list(data)

def calculate_bleu_score(references, hypotheses):
    smoothie = SmoothingFunction().method4
    bleu_scores = [sentence_bleu([ref], hyp, smoothing_function=smoothie) for ref, hyp in zip(references, hypotheses)]
    return sum(bleu_scores) / len(bleu_scores)

test_dataset = load_from_json(f'data/{config.LANGS}-test_data.json')
test_dlp = DataloaderProvider(test_dataset, config.BATCH_SIZE, tokenizer, "dataloader/tokenized_test_dataset.json", load_dataset=False)
test_dataloader = test_dlp.dataloader
loss_function = torch.nn.CrossEntropyLoss(ignore_index=test_dlp.pad)


model.eval()
validation_loss = 0
num_batches = 0
references = []
hypotheses = []

with torch.no_grad():
    for batch in test_dataloader:
        sources = batch['sources'].to(model.device)
        targets = batch['targets'].to(model.device)

        predictions = model(sources, targets[:, :-1])
        B, S, C = predictions.shape

        loss = loss_function(predictions.reshape(-1, C), targets[:, 1:].reshape(-1))
        validation_loss += loss.item()
        num_batches += 1

        pred_tokens = torch.argmax(predictions, dim=-1).cpu().tolist()
        target_tokens = targets[:, 1:].cpu().tolist()


        for pred, target in zip(pred_tokens, target_tokens):
            pred_str = tokenizer.decode(pred, skip_special_tokens=True)
            target_str = tokenizer.decode(target, skip_special_tokens=True)
            hypotheses.append(pred_str)
            references.append(target_str)

avg_validation_loss = validation_loss / num_batches
bleu_score = calculate_bleu_score(references, hypotheses)
print(f"Average Validation Loss: {avg_validation_loss}")
print(f"BLEU Score: {bleu_score}")