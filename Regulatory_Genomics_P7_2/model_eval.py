import torch
from model_architecture import FineTunedSpeciesLM

# 1) Choose device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 2) Re-instantiate your model architecture
model = FineTunedSpeciesLM().to(device)

# 3) Load in the checkpoint
ckpt = torch.load('checkpoints/best_model.pt', map_location=device)

# 4) Populate your model’s parameters
model.load_state_dict(ckpt)

# 5) (Optional) put in eval mode for inference
model.eval()