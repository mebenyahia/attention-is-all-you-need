Download the dataset (needed to run training and evaluation scripts):
1. Adjust the training subset size and language pair in config.py.
2. script -c "python download_dataset.py" log_download_dataset.txt

Configuration (needed to run models from our presentation):
1. Download the corresponding archive from https://drive.google.com/drive/folders/1V5u4u_5HQFyxP8Kp_HC4wWX05qKQpBH0?usp=drive_link) and paste the contents (config.py and the folders accelerator_checkpoint_X, saved_model, tokenizer) to the training folder.
2. Download the correct training split (40000 for config_1 and config_2, 300000 for config_3 and config_4) and save it to training/data/{langs}-train_data.json

Train a tokenizer:
1. Adjust vocabulary size in config.py.
2. script -c "python train_tokenizer.py" log_train_tokenizer.txt

Train a new model:
1. Adjust model dimensions and training parameters in config.py 
2. script -c "python train.py" log_train.txt.

Continiue training after interruption/train existing model:
1. Set 'resume' variable to 'true' in train.py.
2. For accelerator_checkpoint_X set train_count to X+1.
3. Adjust checkpoint_dir in line 83 to the saved checkpoints directory.
2. script -c "python train.py" log_train.txt.

Test/evaluate existing model: 
1. script -c "python evaluate.py" log_evaluate.txt
2. script -c "python test.py" log_test.txt
