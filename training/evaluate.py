import torch
from accelerate import Accelerator
from tokenizers import Tokenizer
from models import Transformer
import config

# Initialize the accelerator
output_dir = f"saved_model"
accelerator = Accelerator()

# Load the tokenizer
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
max_length = 50

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

def print_example(input_text, actual_translation):
    output_text = predict(input_text)
    print(f"Input:  {input_text}")
    print(f"Output: {output_text}")
    print(f"Actual: {actual_translation}")
    print()

# Test the with some input

print_example("The federal tax itself, 18.4 cents per gallon, hasn't gone up in 20 years.", "La taxe fédérale elle-même, qui est de 18,4 cents par gallon, n'a pas augmenté depuis 20 ans.")

print_example("A feast for fans", "Un festin pour ses fans")

print_example("People are paying more directly into what they are getting.", "Les gens paient plus directement pour les avantages qui leur sont procurés.")

print_example("However, Atomica is not the only track to have been released.", "Mais Atomica n'est pas le seul titre à dévoiler ses charmes.")


