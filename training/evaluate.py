import os

import torch
from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration
from tokenizers import Tokenizer
from models import Transformer
import config
import json
import math

# Initialize the accelerator
output_dir = "saved_model"
accelerator = Accelerator()

# Load the trained model
model = Transformer(config.VOCAB_SIZE, config.D_MODEL, config.D_FF, config.N_HEADS, config.N_LAYERS)
model = accelerator.prepare(model)
accelerator.load_state(output_dir)
max_length = 50

# Load the tokenizer
tokenizer = Tokenizer.from_file('tokenizer/bpe.json')


# Function to predict output from input text
def predict(input_text):
    with torch.no_grad():
        model.eval()
        vocab = tokenizer.get_vocab()
        sos = vocab["[START]"]
        end = vocab["[END]"]

            # Tokenize the input text

        tokens = tokenizer.encode(input_text).ids
        sources = [sos] + tokens + [end]
        targets = [sos]
        torch_sources_tokens = torch.tensor(sources, dtype=torch.long).unsqueeze(0).to(model.device)

        for _ in range(max_length):
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
        output_text = tokenizer.decode(targets[1:-1])

        return output_text


# Test the model with some input
input_text = "Hi, how are you?"
output_text = predict(input_text)
print(f"Input: {input_text}")
print(f"Output: {output_text}")