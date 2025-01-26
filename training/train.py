import csv
import time

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

resume = True
train_count = 1 #used for the checkpoint directory name

# only for mac, remove for training on cuda or cpu
if torch.backends.mps.is_available():
    os.environ["ACCELERATE_USE_MPS_DEVICE"] = "True"

print("Loading training set from saved...")

def load_from_json(filename):
    with open(filename, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return Dataset.from_list(data)


train_dataset = load_from_json(f'data/{config.LANGS}-train_data.json')
print(f'Dataset size: {len(train_dataset)}')

# Convert the train_dataset_subset to a list of dictionaries
train_dataset_subset_list = train_dataset_subset.to_dict()

# Define the file path for the JSON file
json_file_path = 'data/de_en_train_dataset_subset.json'

# Save the list of dictionaries to the JSON file
with open(json_file_path, 'w', encoding='utf-8') as json_file:
    json.dump(train_dataset_subset_list, json_file, ensure_ascii=False, indent=4)

print("Loading validation set from saved...")
validation_dataset = load_from_json(f'data/{config.LANGS}-validation_data.json')


print("Loading tokenizer from saved...")
tokenizer = Tokenizer.from_file('tokenizer/bpe.json')
vocab = tokenizer.get_vocab()
pad = vocab["[PAD]"]

train_dlp = DataloaderProvider(train_dataset, config.BATCH_SIZE, tokenizer, "dataloader/tokenized_train_dataset.json", load_dataset=True, lang_1=config.LANG_1, lang_2=config.LANG_2)
validation_dlp = DataloaderProvider(validation_dataset, config.BATCH_SIZE, tokenizer, "dataloader/tokenized_validation_dataset.json", load_dataset=True, lang_1=config.LANG_1, lang_2=config.LANG_2)
validation_dataloader = validation_dlp.dataloader


output_dir = f"accelerator_checkpoints_{train_count}"
accelerator_project_config = ProjectConfiguration(
    total_limit=3,
    automatic_checkpoint_naming=True,
    project_dir=output_dir,
    logging_dir=os.path.join(output_dir, 'logs'),
)

accelerator = Accelerator(project_config=accelerator_project_config)

print("Preparing model...")
model = Transformer(
    vocab_size=config.VOCAB_SIZE, 
    d_model=config.D_MODEL, 
    d_ff=config.D_FF, 
    num_heads=config.N_HEADS, 
    N=config.N_LAYERS, 
    max_seq_len=config.ALLOWED_SEQ_LENGTH, 
    pad_token=pad, 
    seed=config.SEED)
optimizer = torch.optim.Adam(model.parameters(), lr=config.L_RATE, betas=(config.BETA_1, config.BETA_2), eps=config.EPS)
loss_function = torch.nn.CrossEntropyLoss(ignore_index=train_dlp.pad, label_smoothing=config.EPS_LS)
sheduler = ReduceLROnPlateau(optimizer, factor=0.5)

model, optimizer, train_dataloader = accelerator.prepare(model, optimizer, train_dlp.dataloader)
accelerator.register_for_checkpointing(sheduler)
accelerator.register_for_checkpointing(model)
accelerator.register_for_checkpointing(optimizer)

last_batch = 0  # use last batch to continue tr aining from that point

if resume:
    print("Loading accelerator state...")
    previous_run_dir = f"accelerator_checkpoints_{train_count - 1}"
    checkpoint_dir = f"{previous_run_dir}/checkpoints/checkpoint_91" #change the directory name to the desired checkpoint
    accelerator.load_state(checkpoint_dir)
    with open(f"{previous_run_dir}/metadata.json", "r") as f:
        metadata = json.load(f)
    last_batch = metadata["batch"]
    train_dataloader_backup = train_dataloader
    skipped_dataloader = accelerator.skip_first_batches(train_dataloader, last_batch)


def calculate_validation_loss(model, validation_dataloader, loss_function):
    model.eval()
    validation_loss = 0
    num_batches = 0

    with torch.no_grad():
        for batch in validation_dataloader:
            sources = batch['sources'].to(model.device)
            targets = batch['targets'].to(model.device)

            predictions = model(sources, targets[:, :-1])
            B, S, C = predictions.shape

            loss = loss_function(predictions.reshape(-1, C), targets[:, 1:].reshape(-1))
            validation_loss += loss.item()
            num_batches += 1

    avg_validation_loss = validation_loss / num_batches
    return avg_validation_loss

def train_epoch(model, train_dataloader, optimizer, loss_function, sheduler, accelerator):
    model.train()
    train_loss = 0
    train_losses_log = []
    num_batches = last_batch
    ind_num_batches = 0
    
    for batch in train_dataloader:

        sources = batch['sources']
        targets = batch['targets']

        optimizer.zero_grad()
        predictions = model(sources, targets[:, :-1])

        B, S, C = predictions.shape

        loss = loss_function(predictions.reshape(-1, C), targets[:, 1:].reshape(-1))
        accelerator.backward(loss)
        optimizer.step()
        train_loss += loss.item()
        train_losses_log.append(loss.item())
        num_batches += 1
        ind_num_batches += 1
        
        if num_batches % 30 == 0:
            print(f'Batch {num_batches}: training loss {loss.item()}')
        if num_batches % 300 == 0:
            print(f"Saving checkpoint to {output_dir}...")
            accelerator.save_state()
            # Save progress
            metadata = {
                "batch": num_batches,
                "training_losses_epoch": train_losses_log
            }
            with open(f"{output_dir}/metadata.json", "w") as f:
                json.dump(metadata, f)

    avg_train_loss = train_loss / ind_num_batches
    avg_validation_loss = calculate_validation_loss(model, validation_dataloader, loss_function)
    #print(f"Average Validation Loss: {avg_validation_loss}")
    sheduler.step(avg_train_loss)
    return avg_train_loss, avg_validation_loss

try:
    print("Training: ")
    losses = []
    validation_losses = []
    log_file = 'training_log.csv'

    # Write the header if the file does not exist
    if not os.path.exists(log_file):
        with open(log_file, 'w', newline='') as f:
            writer = csv.writer(f, delimiter=';')
            writer.writerow(['Epoch', 'Time', 'Train_Loss', 'Val_Loss'])

    for epoch in range(config.EPOCHS):
        start_time = time.time()
        if epoch == 0 and resume:
            train_dataloader = skipped_dataloader
        elif epoch != 0:
            last_batch = 0
            train_dataloader = train_dataloader_backup
        train_loss, val_loss = train_epoch(model=model, train_dataloader=train_dataloader, optimizer=optimizer, loss_function=loss_function,
                    sheduler=sheduler, accelerator=accelerator)
        print(f'Epoch {epoch + 1}: Average Training Loss: {train_loss}, Average Validation Loss: {val_loss}')
        losses.append(train_loss)
        validation_losses.append(val_loss)
        end_time = time.time()
        epoch_time = end_time - start_time
        # Log the epoch, time, train loss, and validation loss
        with open(log_file, 'a', newline='') as f:
            writer = csv.writer(f, delimiter=';')
            writer.writerow([epoch + 1, f"{epoch_time:.2f}", f"{train_loss:.4f}", f"{val_loss:.4f}"])
finally:
    print("Saving model...")
    accelerator.save_model(model, "saved_model")
