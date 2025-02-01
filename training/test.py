import json
import torch
from accelerate import Accelerator
from datasets import Dataset
from tokenizers import Tokenizer
from models import Transformer, DataloaderProvider
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import config

# Initialize the accelerator
output_dir = f"saved_model"
accelerator = Accelerator()
tokenizer = Tokenizer.from_file(f"tokenizer/bpe.json")
tokenizer.enable_truncation(max_length=config.ALLOWED_SEQ_LENGTH)

vocab = tokenizer.get_vocab()
sos = vocab["[START]"]
end = vocab["[END]"]
pad = vocab["[PAD]"]

# Load the trained model
model = Transformer(
    vocab_size=config.VOCAB_SIZE,
    d_model=config.D_MODEL,
    d_ff=config.D_FF,
    num_heads=config.N_HEADS,
    N=config.N_LAYERS,
    max_seq_len=config.ALLOWED_SEQ_LENGTH,
    pad_token=pad,
    seed=config.SEED)
model = accelerator.prepare(model)
accelerator.load_state(output_dir)

# Utility to load dataset
def load_from_json(filename):
    print("Loading testing set from saved...")
    with open(filename, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return Dataset.from_list(data)

# BLEU Score Calculation
def calculate_bleu_score(references, hypotheses):
    print("Calculating BLEU Score...")
    smoothie = SmoothingFunction().method4
    bleu_scores = [
        sentence_bleu([ref.split()], hyp.split(), smoothing_function=smoothie)
        for ref, hyp in zip(references, hypotheses)
    ]
    return sum(bleu_scores) / len(bleu_scores), bleu_scores


# Load test data
test_dataset = load_from_json(f"data/{config.LANGS}-test_data.json")
test_dlp = DataloaderProvider(test_dataset, config.BATCH_SIZE, tokenizer, f"dataloader/tokenized_test_dataset.json", lang_1=config.LANG_1, lang_2=config.LANG_2, load_dataset=False)
test_dataloader = test_dlp.dataloader

loss_function = torch.nn.CrossEntropyLoss(ignore_index=test_dlp.pad, label_smoothing=config.EPS_LS)

# Evaluate the model
model.eval()
test_loss = 0
num_batches = 0
references = []
hypotheses = []
full_sentence_hypotheses = []
full_sentence_references = []

with torch.no_grad():
    print("Evaluating the model...")
    for batch in test_dataloader:
        sources = batch['sources'].to(model.device)
        targets = batch['targets'].to(model.device)

        print("Generating whole sentence predictions..")
        generated = model.generate(sources, max_length=config.ALLOWED_SEQ_LENGTH, start_token=sos, end_token=end)
        
        generated_seqs = [seq.tolist() for seq in generated]
        decoded_batch = tokenizer.decode_batch(generated_seqs, skip_special_tokens=True)
        full_sentence_hypotheses.extend(decoded_batch)

        batch_references = tokenizer.decode_batch([t.tolist() for t in targets], skip_special_tokens=True)
        full_sentence_references.extend(batch_references)

        print("Calculating loss and token-level prediction..")
        # Calculate token-level loss
        predictions = model(sources, targets[:, :-1])
        B, S, C = predictions.shape
        loss = loss_function(predictions.reshape(-1, C), targets[:, 1:].reshape(-1))
        test_loss += loss.item()
        print(f"Batch {num_batches}: loss {loss.item()}")
        num_batches += 1

        # Decode batch-wise predictions for token-level BLEU
        pred_tokens = torch.argmax(predictions, dim=-1).cpu().tolist()
        target_tokens = targets[:, 1:].cpu().tolist()

        for pred, target in zip(pred_tokens, target_tokens):
            # Last token BLEU
            pred_str = tokenizer.decode(pred, skip_special_tokens=True)
            target_str = tokenizer.decode(target, skip_special_tokens=True)
            hypotheses.append(pred_str)
            references.append(target_str)
        

# Calculate Metrics
avg_test_loss = test_loss / num_batches
token_level_bleu, token_level_bleu_list = calculate_bleu_score(references, hypotheses)
sentence_level_bleu, sentence_level_bleu_list = calculate_bleu_score(full_sentence_references, full_sentence_hypotheses)

# Print Results
print(f"Average test Loss: {avg_test_loss:.4f}")
print(f"Token-Level BLEU Score: {token_level_bleu:.4f}")
print(f"Full-Sentence BLEU Score: {sentence_level_bleu:.4f}")

results = {
    "avg_test_loss": avg_test_loss,
    "avg_token_level_bleu": token_level_bleu,
    "avg_sentence_level_bleu": sentence_level_bleu,
    "token_level_bleu_list": token_level_bleu_list,
    "sentence_level_bleu_list": sentence_level_bleu_list
}
with open(f"results.json", "w") as f:
                json.dump(results, f) 