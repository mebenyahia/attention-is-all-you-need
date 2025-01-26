import os

import torch
from accelerate import Accelerator
from tokenizers import Tokenizer
from models import Transformer
import config


# Initialize the accelerator
output_dir = "saved_model"
accelerator = Accelerator()
tokenizer = Tokenizer.from_file('tokenizer/bpe.json')
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
max_length = 50

# Load the tokenizer

# Function to predict output from input text
def predict(input_text):
    with torch.no_grad():
        model.eval()

            # Tokenize the input text

        tokens = tokenizer.encode(input_text).ids
        sources = [sos] + tokens + [end]
        targets = [sos]
        torch_sources_tokens = torch.tensor(sources, dtype=torch.long).unsqueeze(0).to(model.device)

        for i in range(max_length):
            # Prepare the current output tensor
            torch_targets_tokens = torch.tensor(targets, dtype=torch.long).unsqueeze(0).to(model.device)

            # Forward pass through the model
            logits = model(torch_sources_tokens, torch_targets_tokens)

            # Get the token with the highest probability
            next_token = torch.argmax(logits[:, -1, :], dim=-1).item()
            # Append the token to the output
            targets.append(next_token)

            # Stop if `[END]` token is generated
            if next_token == end:
                break

        # Decode the output tokens
        output_text = tokenizer.decode(targets)
        print(targets)
        return output_text

# Test the with some input
input_text = "The push comes as the country's Highway Trust Fund, financed with taxes Americans pay at the gas pump, is broke."
output_text = predict(input_text)
print(f"Input: {input_text}")
print(f"Output: {output_text}")
print(f"Actual: La pression vient du fait que le Highway Trust Fund du pays, financé avec les taxes que les Américains paient à la pompe, est financièrement à sec.")