Implementations of attention all you need

TODOs for reproduction:
 - Tokenization should be done before the training starts and not in the collate function.
 - Tracking of validation and training loss, the bleu score during training.  (training)
 - Testing the model on the test dataset, computing the loss on the test dataset.  (evaluation)
 - Code for sampling the model.  (evaluation)
 - Code for plots and visualizations.   (evaluation)
 - The scripts that we implement need to run on the TU jupyter cluster which we have access to.  (training and evaluation)