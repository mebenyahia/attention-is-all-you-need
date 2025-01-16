import torch
from accelerate.utils import ProjectConfiguration
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


train_dataset = load_from_json(f'data/{config.LANGS}-train_data.json')

print("Loading tokenizer from saved...")
tokenizer = Tokenizer.from_file('tokenizer/bpe.json')

train_dlp = DataloaderProvider(train_dataset, config.BATCH_SIZE, tokenizer)

output_dir = "accelerator_checkpoint"
accelerator_project_config = ProjectConfiguration(
    total_limit=3,
    automatic_checkpoint_naming=True,
    project_dir=output_dir,
    logging_dir=os.path.join(output_dir, 'logs'),
)

accelerator = Accelerator(project_config=accelerator_project_config)

print("Computing positional encodings...")

print("Preparing model...")
model = Transformer(config.VOCAB_SIZE, config.D_MODEL, config.D_FF, config.N_HEADS, config.N_LAYERS)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-5, betas=(0.9, 0.98), eps=1e-9)
loss_function = torch.nn.CrossEntropyLoss()
sheduler = ReduceLROnPlateau(optimizer, factor=0.5)

model, optimizer, train_dataloader = accelerator.prepare(model, optimizer, train_dlp.dataloader)
accelerator.register_for_checkpointing(sheduler)
accelerator.register_for_checkpointing(model)
accelerator.register_for_checkpointing(optimizer)

output_dir = "accelerator_checkpoints"
last_batch = 0 #use last batch to continue training from that point
if resume:
    print("Loading accelerator state...")
    accelerator.load_state()
    with open(f"{output_dir}/metadata.json", "r") as f:
        metadata = json.load(f)
    last_batch = metadata["batch"]
    skipped_dataloader = accelerator.skip_first_batches(train_dataloader, last_batch)


if not resume:
    print("Saving initial accelerator state...")
    accelerator.save_state()


def train_epoch(model, train_dataloader, optimizer, loss_function, sheduler, accelerator):
    model.train()
    train_loss = 0
    num_batches = last_batch


    for batch in train_dataloader:

        sources = batch['sources']
        targets = batch['targets']

        optimizer.zero_grad()
        predictions = model(sources, targets[:, :-1])

        B, S, C = predictions.shape

        loss = loss_function(predictions.reshape(-1, C), targets[:, 1:].reshape(-1))
        # loss.backward()
        accelerator.backward(loss)
        optimizer.step()
         
        # TODO: calculate validation loss
        print(f'training loss: {loss.item()}, validation Loss')
        train_loss += loss.item()
        num_batches += 1


        if num_batches % 10 == 0:
            checkpoint_dir = f"accelerator_checkpoints"
            print(f"Saving checkpoint to {checkpoint_dir}...")
            accelerator.save_state()
            # Save progress
            metadata = {
                "batch": num_batches,
            }
            with open(f"{checkpoint_dir}/metadata.json", "w") as f:
                json.dump(metadata, f)

        if num_batches == 200:
            break
        sheduler.step(loss.item())
        



print("Training: ")

for epoch in range(config.EPOCHS):
    train_epoch(model=model, train_dataloader=train_dataloader, optimizer=optimizer, loss_function=loss_function,
            sheduler=sheduler, accelerator=accelerator)
    
    

