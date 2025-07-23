import torch
import torch.nn as nn
import torch.nn.functional as F

class CrossEntropyNucleotideLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.criterion = nn.CrossEntropyLoss(reduction="none")

    def forward(self, nuc_logits, one_hot_labels):
        B, L = nuc_logits.shape[:2]
        """
        Args:
            nuc_logits: Tensor of shape (B, 2003, 4) — raw logits for A/C/G/T
            one_hot_labels: Tensor of shape (B, 2003, 4) — one-hot encoded labels
        
        Returns:
            scalar loss tensor
            edit kay: added reduction = "none"; to maintain shape, to be able to apply loss mask
        """
        targets = one_hot_labels.argmax(dim=-1)        # shape: (B, 2003)
        logits_flat = nuc_logits.reshape(-1, 4)           # shape: (B*2003, 4)
        targets_flat = targets.view(-1)                # shape: (B*2003,)

        loss_flat = self.criterion(logits_flat, targets_flat) # shape: (B*2003,)
        loss = loss_flat.view(B, L)                     # shape: (B, 2003)

        return loss

    def backward(self, loss):
        """
        Manually call backward on the computed loss.
        """
        loss.backward()




class VariationAndAltAlleleLoss(nn.Module):
    def __init__(self, lambda_var=1.0, lambda_alt=1.0, epsilon=1e-8):
        super().__init__()
        self.lambda_var = lambda_var
        self.lambda_alt = lambda_alt
        self.epsilon = epsilon

    # TODO: reference nucleotide calculation

    # TODO: neutral model extraction and correction

    def forward(self, nuc_logits, target_idx, ref_idx):
        """
        Args:
            nuc_logits: Tensor of shape (B, L, 4)
            target_idx: LongTensor of shape (B, L), true nucleotide index
            ref_idx: LongTensor of shape (B, L), reference base index

        Returns:
            scalar loss
        """
        B, L, C = nuc_logits.shape
        probs = F.softmax(nuc_logits, dim=-1)  # shape (B, L, 4)

        # --- Binary Cross Entropy for Variation ---
        p_ref = probs.gather(-1, ref_idx.unsqueeze(-1)).squeeze(-1)  # (B, L)
        p_var = 1.0 - p_ref                                          # probability of having variation (B, L)
        is_variant = (target_idx != ref_idx).float()  # binary ground truth (if the target is different from the reference → True)

        bce_loss = F.binary_cross_entropy(p_var, is_variant, reduction='mean')  # scalar

        # --- Cross Entropy for Allele Prediction (only at variant sites) ---
        is_var_mask = (is_variant == 1.0) # boolean mask that selects positions where there is a variant

        if is_var_mask.sum() > 0:
            # select only variant positions
            selected_logits = nuc_logits[is_var_mask]         # shape (N_var, 4)
            selected_targets = target_idx[is_var_mask]        # shape (N_var,)
            ce_loss = F.cross_entropy(selected_logits, selected_targets, reduction='mean')
        else:
            ce_loss = torch.tensor(0.0, device=nuc_logits.device)

        total_loss = self.lambda_var * bce_loss + self.lambda_alt * ce_loss
        return total_loss
