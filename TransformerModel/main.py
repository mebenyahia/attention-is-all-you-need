import argparse
from torch.utils.data import DataLoader
from models.transformer import TransformerModel
from datasets import WMT14Dataset
from training.train_model import train, evaluate
from training.utils import create_optimizer, setup_tensorboard, save_checkpoint, load_checkpoint

def main(args):
    # Model configuration
    ntokens = 1000  # size of the vocabulary
    ninp = 512      # embedding dimension
    nhead = 8       # number of heads in the multi-head attention models
    nhid = 2048     # dimension of the feedforward network model in nn.TransformerEncoder
    nlayers = 6     # number of nn.TransformerEncoderLayer in nn.TransformerEncoder
    dropout = 0.2   # dropout probability

    # Initialize the model
    model = TransformerModel(ntokens, ninp, nhead, nhid, nlayers, dropout)
    
    if args.mode == 'train':
        # Training setup
        train_dataset = WMT14Dataset(filepath=args.train_data)
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=train_dataset.collate_fn)
        optimizer = create_optimizer(model, lr=0.005)
        criterion = torch.nn.CrossEntropyLoss()
        writer = setup_tensorboard('runs/Transformer')

        # Training loop
        for epoch in range(1, args.epochs + 1):
            train_loss = train(model, train_loader, optimizer, criterion, epoch, writer)
            print(f'Epoch {epoch}: Training Loss {train_loss:.4f}')
            writer.add_scalar('Training Loss', train_loss, epoch)

        writer.close()
        save_checkpoint({'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict()}, filename='final_model.pth')

    elif args.mode == 'evaluate':
        # Evaluation setup
        test_dataset = WMT14Dataset(filepath=args.test_data)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=test_dataset.collate_fn)
        criterion = torch.nn.CrossEntropyLoss()

        # Load model from checkpoint
        checkpoint_path = 'final_model.pth'
        model, _, _ = load_checkpoint(model, None, checkpoint_path)
        model.eval()

        # Evaluation loop
        test_loss = evaluate(model, test_loader, criterion)
        print(f'Final Evaluation Loss: {test_loss:.4f}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train or Evaluate the Transformer Model')
    parser.add_argument('--mode', choices=['train', 'evaluate'], required=True, help='Mode to run the script in')
    parser.add_argument('--train_data', type=str, help='Filepath for training data')
    parser.add_argument('--test_data', type=str, help='Filepath for test data')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for training or evaluation')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs to train')
    args = parser.parse_args()

    main(args)
