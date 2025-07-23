import torch

class LossMask:
    def __init__(self, sequence_length=2003, train_ratio=0.8):
        self.seq_len = sequence_length
        self.train_mask_1d, self.val_mask_1d = self.create_position_masks(train_ratio)

    def create_position_masks(self, train_ratio=0.8):
        torch.manual_seed(5)
        perm = torch.randperm(self.seq_len)
        n_train = int(train_ratio * self.seq_len)
        train_mask = torch.zeros(self.seq_len, dtype=torch.float32)
        val_mask = torch.zeros(self.seq_len, dtype=torch.float32)
        train_mask[perm[:n_train]] = 1.0
        val_mask[perm[n_train:]] = 1.0
        return train_mask, val_mask

    def get_mask(self, batch_size, mode, device):
        mask_1d = self.train_mask_1d if mode == 'train' else self.val_mask_1d
        return mask_1d.to(device).unsqueeze(0).expand(batch_size, -1)  # shape: (B, seq_len)

    def apply_mask(self, loss_tensor, mode):
        """
        Args:
            loss_tensor: Tensor of shape (B, seq_len) — per-position loss
            mode: 'train' or 'val'
        Returns:
            scalar average masked loss
        """
        B, L = loss_tensor.shape
        device = loss_tensor.device
        mask = self.get_mask(B, mode=mode, device=device)  # shape: (B, seq_len)

        masked_loss = loss_tensor * mask  # zero out non-target positions

        # Sum masked loss
        total_loss = masked_loss.sum()

        # Count number of contributing elements
        token_count = mask.sum().clamp(min=1)  # avoid division by zero

        # Return average
        average_loss = total_loss / token_count
        return average_loss
