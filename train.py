import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm 
import time
from utils import save
from sklearn.preprocessing import MinMaxScaler

class Trainer:
    def __init__(self, config, train_dataset, eval_dataset, test_dataset):
        """
        Initialize the Trainer class.

        Args:
            config (object): Configuration object containing model and training parameters.
            train_dataset (Dataset): Training dataset.
            eval_dataset (Dataset): Evaluation dataset.
            test_dataset (Dataset): Test dataset.
        """

        torch.manual_seed(8)

        # Assign attributes from the configuration and input datasets
        self.config = config
        # self.model = config.model
        self.loss_fn = config.loss_fn
        self.epochs = config.epochs

        # Assign dataset
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.test_dataset = test_dataset
        self.train_dataloader = self.get_train_dataloader(
            train_dataset = self.train_dataset,
            batch_size = self.config.batch_size,
        )
        self.eval_dataloader = self.get_eval_dataloader(
            eval_dataset = self.eval_dataset,
            batch_size = self.config.batch_size,
        )
        self.test_dataloader = self.get_test_dataloader(
            test_dataset = self.test_dataset,
            batch_size = 100,
        )

        # Initialize the optimizer with the model parameters and learning rate
        # self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config.learning_rate)


    def get_train_dataloader(self, train_dataset, batch_size, shuffle=True):
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)
        return train_dataloader

    def get_eval_dataloader(self, eval_dataset, batch_size, shuffle=False):
        eval_dataloader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=shuffle)
        return eval_dataloader

    def get_test_dataloader(self, test_dataset, batch_size, shuffle=False):
        test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=shuffle)
        return test_dataloader

    def training_step(self, model, optimizer, patience=10):
        """
        Perform one training step over multiple epochs.

        Args:
            patience (int): Number of epochs to wait for improvement in validation loss before early stopping.

        Yields:
            dict: Dictionary containing information about each epoch, including training and validation losses.
                - epoch: Current epoch number
                - training_losses: List of training losses over epochs
                - validation_losses: List of validation losses over epochs
                - epoch_loss_str: Formatted string representing current epoch's loss
                - final: Whether it's the final epoch (True/False)
        """
        training_losses = list()
        validation_losses = list()

        # Training loop
        epoch_loss = ""
        estimated_time = ""
        progress_bar = tqdm(total=self.epochs, desc="Training Progress", dynamic_ncols=True)

        # Set stopping criteria
        best_val_loss = float('inf')
        current_patience = 0
        final = False

        

        for epoch in range(self.epochs):

            model.train()
            t_start = time.time()
            batch_losses = list()
            
            for batch, (X, y) in enumerate(self.train_dataloader):
                
                # Preprocess input and target tensors
                X = X.permute(1, 0, 2)
                y = torch.unsqueeze(y, 1)
                X, y = X.to(self.config.device).to(torch.float32), y.to(self.config.device).to(torch.float32)
                
                y_pred = model(X)

                # Calculate and record the batch loss
                loss = self.loss_fn(y_pred, y)
                batch_losses.append(loss.item())

                # Backward pass and optimization step
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            # Calculate  average loss over sequences and batches
            epoch_loss = np.mean(batch_losses)
            training_losses.append(epoch_loss)

            val_loss = self.evaluate(
                model = model,
                eval_dataloader = self.eval_dataloader
            )
            validation_losses.append(val_loss)

            # Format text for display
            format_text = lambda t: f"{t:.3f}" if t<10**3 and t>=10**-3 else f"{t:.3e}"
            epoch_loss = f", Loss: {format_text(epoch_loss)}"
            estimated_time = ", ETC: %.2f minutes (%.2f seconds)" %\
                    ((self.epochs - epoch - 1) * (time.time() - t_start) / 60,
                    (self.epochs - epoch - 1) * (time.time() - t_start))
            
            progress_bar.set_postfix_str(f"Epoch: {epoch+1}/{self.epochs}{epoch_loss}{estimated_time}")
            progress_bar.update(1)

            # Check for improvement in validation loss
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                current_patience = 0
            else:
                current_patience += 1
            
            # Early stopping check
            if current_patience >= patience:
                print(f"Early stopping at epoch {epoch+1} due to no improvement in validation loss.")
                final = True
                break

            if epoch + 1 == self.epochs:
                print(f"Training completed for all {self.epochs} epochs.")
                final = True

            # Yield information about the current epoch
            yield {
                "epoch": epoch, 
                "training_losses": training_losses, 
                "validation_losses": validation_losses,
                "epoch_loss_str": epoch_loss, 
                "estimated_time": estimated_time, 
                "final": final,
            }
            
        progress_bar.close()

    def train(self, model, optimizer):

        model.train()
        # Iterate over training steps
        for result in self.training_step(model, optimizer):
            training_losses = result['training_losses']
            validation_losses = result['validation_losses']
            is_final = result['final']

        return model, training_losses, validation_losses

    def evaluate(self, model, eval_dataloader):
        model.eval()
        inputs, targets = next(iter(eval_dataloader))

        # Preprocess input and target tensors
        inputs = inputs.permute(1, 0, 2).to(self.config.device).to(torch.float32)
        targets = torch.unsqueeze(targets, 1).to(self.config.device).to(torch.float32)

        # Perform forward pass without gradient computation
        with torch.no_grad():
            outputs = model(inputs)
            val_loss = self.loss_fn(outputs, targets)

        return val_loss.item()

    def predict(self, forecast_model):

        forecast_model.eval()
        inputs, targets = next(iter(self.test_dataloader))
        
        # Preprocess input and target tensors
        inputs = inputs.permute(1, 0, 2)
        inputs = inputs.to(self.config.device).to(torch.float32)
        targets = torch.unsqueeze(targets, 1)
        targets = targets.to(self.config.device).to(torch.float32)

        # Perform forward pass without gradient computation
        with torch.no_grad():
            outputs = forecast_model(inputs)

        return outputs, targets