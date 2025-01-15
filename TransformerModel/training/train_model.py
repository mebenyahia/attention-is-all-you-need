import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from models.transformer import TransformerModel
from datasets import WMT14Dataset  # Make sure this import matches the location and name of your dataset class
from training.utils import save_checkpoint, load_checkpoint, create_optimizer, setup_tensorboard

def train(model, data_loader, optimizer, criterion, epoch, writer):
    model.train()
    total_loss = 0
    for batch_idx, (src, trg) in enumerate(data_loader):
        optimizer.zero_grad()
        output = model(src)
        loss = criterion(output, trg)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        if batch_idx % 100 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(src), len(data_loader.dataset),
                100. * batch_idx / len(data_loader), loss.item()))
            writer.add_scalar('training_loss', loss.item(), epoch * len(data_loader) + batch_idx)
    
    return total_loss / len(data_loader)

def evaluate(model, data_loader, criterion):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for src, trg in data_loader:
            output = model(src)
            total_loss += criterion(output, trg).item()

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
    epochs = 10
    lr = 0.005      # learning rate

    # Load data
    train_dataset = WMT14Dataset(filepath='data/train/wmt14_translate_fr-en_train.csv')
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=train_dataset.collate_fn)

    test_dataset = WMT14Dataset(filepath='data/test/wmt14_translate_fr-en_test.csv')
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=test_dataset.collate_fn)

    # Initialize the model
    model = TransformerModel(ntokens, ninp, nhead, nhid, nlayers, dropout)
    optimizer = create_optimizer(model, lr)
    criterion = nn.CrossEntropyLoss()

    writer = setup_tensorboard('runs/Transformer')

    best_test_loss = float('inf')
    for epoch in range(1, epochs + 1):
        train_loss = train(model, train_loader, optimizer, criterion, epoch, writer)
        test_loss = evaluate(model, test_loader, criterion)
        writer.add_scalar('test_loss', test_loss, epoch)

        # Save the current model (checkpoint)
        is_best = test_loss < best_test_loss
        best_test_loss = min(test_loss, best_test_loss)
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
        }, is_best)

    writer.close()

if __name__ == '__main__':
    main()
