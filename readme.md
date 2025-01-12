Implementations of attention all you need

TODOs for reproduction:
 - Positional encoding computation should be implemented as part of the model. Either in the Embedding class (with register_buffer()) or as a separate class.  (model)
 - Multi Head Attention has to implemented with broadcasting instead of a for loop.  (model)
 - Implement training loop with checkpoints, the model should be saved every couple of steps (50-100?). If the training is interrupted for some reason we should be able to resume the training from a certain checkpoint.  (training)
 - Tracking of validation and training loss, the bleu score during training.  (training)
 - Testing the model on the test dataset, computing the loss on the test dataset.  (evaluation)
 - Code for sampling the model.  (evaluation)
 - Code for plots and visualizations.   (evaluation)
 - The scripts that we implement need to run on the TU jupyter cluster which we have access to.  (training and evaluation)