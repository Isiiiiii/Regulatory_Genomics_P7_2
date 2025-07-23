import os
import numpy as np
import loss_mask
import torch
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from loss_function import CrossEntropyNucleotideLoss
#from loss_mask import apply_mask
from model_architecture import FineTunedSpeciesLM
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from loss_mask import LossMask
from proj_loader import prepare_data_loader
import numpy as np



def train_fine_tuned_model(
    model,
    data_loader,
    device,
    lossMask,
    lr=1e-4,
    epochs=25,
    patience=5,
    checkpoint_dir='checkpoints_positional',
    checkpoint_name='best_model.pt'
):
    """
    Trains FineTunedSpeciesLM with early stopping and TensorBoard logging using one-hot labels.
    """
    os.makedirs(checkpoint_dir, exist_ok=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    tb_writer = SummaryWriter(log_dir=checkpoint_dir)

    loss_fn = CrossEntropyNucleotideLoss()
    best_val_loss = float('inf')
    epochs_no_improve = 0
    model.to(device)

    for epoch in range(1, epochs + 1):
        # --- Training ---
        model.train()
        train_losses = []
        for batch in tqdm(data_loader, desc=f"Epoch {epoch}/{epochs} [train]", ncols=120, leave=False):
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)  # shape: [B, 2003, 4] (one-hot)

            optimizer.zero_grad()
            outputs = model(input_ids=input_ids)  # shape: [B, 2003, 4]
            pre_loss = loss_fn(outputs, labels)   # shape: [B, 2003]

            #apply loss_mask
            loss = lossMask.apply_mask(pre_loss, "train") # shape: [B, 2003]

            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        avg_train_loss = np.mean(train_losses)
        tb_writer.add_scalar("train/loss", avg_train_loss, epoch)

        # --- Validation ---
        model.eval()
        val_losses = []
        with torch.no_grad():
            for batch in tqdm(data_loader, desc=f"Epoch {epoch}/{epochs} [val]", ncols=120, leave=False):
                input_ids = batch["input_ids"].to(device)
                labels = batch["labels"].to(device)


                outputs = model(input_ids=input_ids)
                pre_val_loss = loss_fn(outputs, labels)

                #apply loss_mask
                val_loss = lossMask.apply_mask(pre_val_loss, "val") # shape: [B, 2003]

                val_losses.append(val_loss.item())

        avg_val_loss = np.mean(val_losses)
        tb_writer.add_scalar("val/loss", avg_val_loss, epoch)

        print(f"Epoch {epoch}: train_loss = {avg_train_loss:.4f}, val_loss = {avg_val_loss:.4f}")

        # --- Checkpoint & Early Stopping ---
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_no_improve = 0
            path = os.path.join(checkpoint_dir, checkpoint_name)
            torch.save(model.state_dict(), path)
            print(f"  ↳ Saved best model at {path}")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"  ↳ Early stopping after {patience} epochs without improvement.")
                break

    tb_writer.close()
    model.load_state_dict(torch.load(os.path.join(checkpoint_dir, checkpoint_name)))
    return model

# 1) Choose device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# The model
model = FineTunedSpeciesLM()

# Load in the checkpoint
checkpoint_path = 'checkpoints_positional/best_model.pt'
if os.path.exists(checkpoint_path):
    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt)
    print(f"Loaded checkpoint from {checkpoint_path}")
else:
    print(f"No checkpoint found at {checkpoint_path}, training from scratch.")
    torch.save(model.state_dict(), checkpoint_path)
    ckpt = torch.load(checkpoint_path, map_location=device)

# Populate your model’s parameters
model.load_state_dict(ckpt)

# Initialize tokenizer and dataset
data_loader = prepare_data_loader(split_positional=True, batch_size=256)

# Loss mask
loss_mask = LossMask()


trained_model = train_fine_tuned_model(
    model=model, 
    data_loader=data_loader, 
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu"), 
    lossMask=loss_mask)