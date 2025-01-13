from datasets import load_dataset
import config
import json
import os

print("Downloading dataset...")

def save_to_json(dataset_split, filename):
    with open(filename, 'w', encoding='utf-8') as f:
        # Convert dataset to a list of dictionaries and save to JSON
        json.dump([example for example in dataset_split], f, ensure_ascii=False, indent=4)

os.makedirs('data', exist_ok=True)

langs = config.LANGS
dataset = load_dataset("wmt14", langs)

# save train, validation, or test splits locally in json format
print("Saving training split...")
save_to_json(dataset['train'], f'data/{langs}-train_data.json')

print("Saving validation split...")
save_to_json(dataset['validation'], f'data/{langs}-validation_data.json')

print("Saving test split...")
save_to_json(dataset['test'], f'data/{langs}-test_data.json')

print("Finished.")
