import torch
from torch.utils.data import DataLoader
from models.transformer import TransformerModel
from datasets import WMT14Dataset  # Import the custom dataset class
from training.utils import load_checkpoint

def evaluate(model, data_loader, criterion):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for src, trg in data_loader:
            output = model(src)
            loss = criterion(output, trg)
            total_loss += loss.item()

    average_loss = total_loss / len(data_loader)
    print('\nTest set: Average loss: {:.4f}\n'.format(average_loss))
    return average_loss

def main():
    # Set the random seed for reproducibility
    torch.manual_seed(42)

    # Hyperparameters
    ntokens = 1000  # size of the vocabulary
    ninp = 512      # embedding dimension
    nhead = 8       # number of heads in the multi-head attention models
    nhid = 2048     # dimension of the feedforward network model in nn.TransformerEncoder
    nlayers = 6     # number of nn.TransformerEncoderLayer in nn.TransformerEncoder
    dropout = 0.2   # dropout probability
    batch_size = 64

    # Initialize the model
    model = TransformerModel(ntokens, ninp, nhead, nhid, nlayers, dropout)
    criterion = torch.nn.CrossEntropyLoss()

    # Load the best saved model checkpoint
    checkpoint_path = 'final_model.pth'
    model, optimizer, start_epoch = load_checkpoint(model, None, checkpoint_path)
    model.eval()  # Make sure to set the model to evaluation mode

    # Load data
    test_dataset = WMT14Dataset(filepath='data/test/wmt14_translate_fr-en_test.csv')
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=test_dataset.collate_fn)

    # Evaluate the model
    test_loss = evaluate(model, test_loader, criterion)
    print(f'Final Test Loss: {test_loss:.4f}')

if __name__ == '__main__':
    main()
