# trainer.py
import time
import torch
import torch.nn.functional as F
from torch.optim import AdamW

class GDTrainer:
    def __init__(self, model, train_loader, val_loader, config, device):
        """
        Args:
            model: An instance of the GDModel.
            train_loader: A PyTorch DataLoader for the training dataset.
            val_loader: A PyTorch DataLoader for the validation dataset.
            config: A configuration dictionary (with keys "training" and "model").
            device: torch.device (e.g., 'cuda' or 'cpu')
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = device

        # Use AdamW (similar to GPT training)
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=config["training"]["learning_rate"],
            weight_decay=config["training"]["weight_decay"]
        )
        self.num_epochs = config["training"]["num_epochs"]

    def evaluate(self, data_loader):
        """
        Evaluates the model on a given dataset.
        Shifts the input tokens so that predictions are aligned with targets.
        """
        self.model.eval()
        total_loss = 0.0
        total_batches = 0
        with torch.no_grad():
            for batch in data_loader:
                # Expecting batch to have "input_ids" of shape (B, L)
                input_ids = batch["input_ids"].to(self.device)
                # For language modeling, we shift the sequence:
                # Use the first L-1 tokens as input and tokens 2...L as targets.
                if input_ids.size(1) < 2:
                    continue
                x = input_ids[:, :-1]      # (B, L-1)
                targets = input_ids[:, 1:]   # (B, L-1)

                # Forward pass: our model returns logits and loss when targets is provided.
                _, loss = self.model(x, targets=targets)
                total_loss += loss.item()
                total_batches += 1
        return total_loss / total_batches if total_batches > 0 else 0.0

    def train(self):
        """
        Runs the training loop for the specified number of epochs.
        Displays training and validation loss as well as elapsed and remaining time.
        Saves the best model (by validation loss) to "best_model.pt".
        """
        self.model.train()
        total_steps = self.num_epochs * len(self.train_loader)
        start_time = time.time()
        current_step = 0
        best_val_loss = float("inf")

        for epoch in range(1, self.num_epochs + 1):
            epoch_loss = 0.0
            for batch in self.train_loader:
                # Expecting batch["input_ids"] to be of shape (B, L)
                input_ids = batch["input_ids"].to(self.device)
                if input_ids.size(1) < 2:
                    continue  # Skip too-short sequences.
                # Shift tokens: input x and targets as described.
                x = input_ids[:, :-1]       # Input: tokens 0 ... L-2
                targets = input_ids[:, 1:]    # Targets: tokens 1 ... L-1

                self.optimizer.zero_grad()
                _, loss = self.model(x, targets=targets)
                loss.backward()
                self.optimizer.step()

                epoch_loss += loss.item()
                current_step += 1

                # Compute ETA
                elapsed = time.time() - start_time
                avg_step_time = elapsed / current_step
                remaining_steps = total_steps - current_step
                eta = remaining_steps * avg_step_time

                print(f"Epoch {epoch}, Step {current_step}/{total_steps}, Loss: {loss.item():.4f}, ETA: {eta/60:.2f} min", end="\r")

            avg_train_loss = epoch_loss / len(self.train_loader)
            val_loss = self.evaluate(self.val_loader)
            print(f"\nEpoch {epoch} complete. Train Loss: {avg_train_loss:.4f}, Val Loss: {val_loss:.4f}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(self.model.state_dict(), "best_model.pt")
                print(f"Saved new best model with Val Loss: {best_val_loss:.4f}")

        total_time = time.time() - start_time
        print(f"Training complete in {total_time/60:.2f} minutes.")
