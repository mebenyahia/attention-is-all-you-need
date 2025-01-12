import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
from accelerate import Accelerator
from datasets import Dataset
from models import Transformer, DataloaderProvider
from tokenizers import Tokenizer
import config
import os
import json
import math

resume = False

# only for mac, remove for training on cuda or cpu
if torch.backends.mps.is_available():
    os.environ["ACCELERATE_USE_MPS_DEVICE"] = "True"


print("Loading training set from saved...")
def load_from_json(filename):
    with open(filename, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return Dataset.from_list(data)

train_dataset = load_from_json(f'{config.LANGS}-train_data.json')

print("Loading tokenizer from saved...")
tokenizer = Tokenizer.from_file('tokenizer/bpe.json')


train_dlp = DataloaderProvider(train_dataset, config.BATCH_SIZE, tokenizer)

accelerator = Accelerator(project_dir="accelerator")

output_dir = "accelerator"
if resume:
    print("Loading accelerator state...")
    accelerator.load_state(output_dir)

print("Computing positional encodings...")
# calculating the positional encoding
# TODO: move this part to the model into the embedding class or as a separate layer
def gen_pos_enc(seq_len, d_model, n):
    pos_enc = torch.zeros((seq_len, d_model), dtype=torch.float)
    
    for k in range(seq_len):
        for i in range(d_model//2):
            theta = k / (n ** (2*i / d_model))
            pos_enc[k, 2*i] = math.sin(theta)
            pos_enc[k, 2*i+1] = math.cos(theta)
            
    pos_enc.requires_grad=False
    return pos_enc

pos_enc = gen_pos_enc(config.ALLOWED_SEQ_LENGTH, config.D_MODEL, 10000)

pos_enc = pos_enc.to('cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu'))

print("Preparing model...")
model = Transformer(config.VOCAB_SIZE, config.D_MODEL, config.D_FF, pos_enc, config.N_HEADS, config.N_LAYERS)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-5, betas=(0.9, 0.98), eps=1e-9)
loss_function = torch.nn.CrossEntropyLoss()
sheduler = ReduceLROnPlateau(optimizer, factor=0.5)

model, optimizer, train_dataloader = accelerator.prepare(model, optimizer, train_dlp.datloader)
accelerator.register_for_checkpointing(sheduler)

if not resume:
    print("Saving initial accelerator state...")
    accelerator.save_state(output_dir="accelerator")

def train_epoch(model, train_dataloader, optimizer, loss_function, sheduler, accelerator):
    model.train()
    train_loss = 0
    num_batches = 0

    for batch in train_dataloader:
        sources = batch['sources']
        targets = batch['targets']
        
        optimizer.zero_grad()
        predictions = model(sources, targets[:, :-1]) 
        
        B, S, C = predictions.shape
        
        loss = loss_function(predictions.reshape(-1, C), targets[:, 1:].reshape(-1))
        #loss.backward()
        accelerator.backward(loss)
        optimizer.step()
                
        print(f'training loss: {loss.item()}')
        train_loss += loss.item()
        num_batches += 1
        
        accelerator.save_state(output_dir="accelerator")
        break
        
    sheduler.step()

print("Training: ")
train_epoch(model=model, train_dataloader=train_dataloader, optimizer=optimizer, loss_function=loss_function, sheduler=sheduler, accelerator=accelerator)

