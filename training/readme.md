Configuration:
1. Move the config.py corresponding to the model that you want to run to the training folder.
   (the configurations that we trained are in the configs folder)

Download the dataset:
1. Adjust the training subset size and language pair in config.py.
2. script -c "python download_dataset.py" log_download_dataset.txt

Train a tokenizer:
1. Adjust vocabulary size in config.py.
2. script -c "python train_tokenizer.py" log_train_tokenizer.txt

Train a model:
1. Adjust model dimensions and training parameters in config.py
2. script -c "python train.py" log_train.txt

Configuaration for evaluation: 
1. Mover the config.py corresponding to the model that you want to run to the training folder.
2. Move copy the correct model to the saved_model folder.
3. Move the tokenizer used for this model to the tokenizer folder.
4. script -c "python evaluate.py" log_evaluate.txt
5. script -c "python test.py" log_test.txt


Download the training set and create a subset.