import torch 
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, random_split
import argparse
from dataset import TextDataset, collate_fn
import sentencepiece as spm
import matplotlib.pyplot as plt
from pathlib import Path
import os
from datetime import datetime

from model_lstm import LSTMLanguageModel
from model_rnn import RNNLanguageModel
from model_transformer import TransformerLanguageModel

# Setting training parameters
BATCH_SIZE = 128
EPOCHS = 30
LR = 1e-3
MAX_LEN = 128

# Registering supported model types
MODEL_REGISTRY = {
    "lstm": LSTMLanguageModel,
    "rnn": RNNLanguageModel,
    "transformer": TransformerLanguageModel
}

def train(model_type: str):
    # Verifying that the selected model is supported
    assert model_type in MODEL_REGISTRY, f"Unsupported model: {model_type}"

    # Selecting the appropriate device (GPU or CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Creating a directory to save results and models
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = Path(f"results/{model_type}_{timestamp}")
    save_dir.mkdir(parents=True, exist_ok=True)

    # Loading the SentencePiece tokenizer model
    sp = spm.SentencePieceProcessor()
    sp.load("tokenizer/spm.model")
    vocab_size = sp.get_piece_size()

    # Loading the dataset and performing an 80/20 split for training and validation
    full_ds = TextDataset("data/train.jsonl", "tokenizer/spm.model", max_len=MAX_LEN)
    train_size = int(0.8 * len(full_ds))
    val_size = len(full_ds) - train_size
    train_ds, val_ds = random_split(full_ds, [train_size, val_size])
    
    # Creating data loaders for batching and shuffling
    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    val_dl = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

    # Initializing the model and optimizer
    model = MODEL_REGISTRY[model_type](vocab_size).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    
    # Setting up learning rate scheduler to reduce LR on plateau of validation loss
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=1, verbose=True)
    
    # Defining the loss function and ignoring padding token (ID=0)
    criterion = nn.CrossEntropyLoss(ignore_index=0)

    # Initializing lists for tracking loss and early stopping
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    epochs_no_improve = 0
    patience = 3  # Patience for early stopping

    # Starting the training loop
    for epoch in range(1, EPOCHS + 1):
        model.train()  # Setting model to training mode
        total_train_loss = 0
        
        # Training phase: iterating over training batches
        for inputs, targets in train_dl:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()  # Clearing previous gradients
            logits, _ = model(inputs)  # Forward passing
            loss = criterion(logits.view(-1, vocab_size), targets.view(-1))  # Computing loss
            loss.backward()  # Backpropagating
            optimizer.step()  # Updating weights
            total_train_loss += loss.item()

        # Averaging training loss
        avg_train_loss = total_train_loss / len(train_dl)
        train_losses.append(avg_train_loss)

        # Validation phase: disabling gradient computation
        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for inputs, targets in val_dl:
                inputs, targets = inputs.to(device), targets.to(device)
                logits, _ = model(inputs)
                loss = criterion(logits.view(-1, vocab_size), targets.view(-1))
                total_val_loss += loss.item()

        # Averaging validation loss
        avg_val_loss = total_val_loss / len(val_dl)
        val_losses.append(avg_val_loss)

        # Updating the learning rate scheduler
        scheduler.step(avg_val_loss)
        current_lr = optimizer.param_groups[0]['lr']

        # Checking for improvement and applying early stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_no_improve = 0
            # Saving the best model based on validation loss
            torch.save(model.state_dict(), save_dir / f"best_{model_type}_model.pt")
        else:
            epochs_no_improve += 1
            if epochs_no_improve == patience:
                print(f"Early stopping at epoch {epoch}")
                break

        # Printing progress
        print(f"Epoch {epoch}/{EPOCHS} - Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, LR: {current_lr:.6f}")

    # Saving the final model after training is complete
    model_save_path = save_dir / f"final_{model_type}_model.pt"
    torch.save(model.state_dict(), model_save_path)
    print(f"✅ Model saved to {model_save_path}")

    # Plotting and saving training and validation loss curves
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'{model_type.upper()} Training & Validation Loss')
    plt.legend()
    plt.grid(True)
    
    plot_path = save_dir / 'loss_curves.png'
    plt.savefig(plot_path)
    print(f"📊 Loss curves saved to {plot_path}")
    plt.close()

# Parsing command-line arguments and launching training
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, choices=["lstm", "rnn", "transformer"], required=True,
                        help="Which model to train: lstm, rnn or transformer")
    args = parser.parse_args()
    train(args.model)
