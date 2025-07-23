import torch
from model_architecture import FineTunedSpeciesLM
import numpy as np
import pandas as pd
from tqdm.auto import tqdm



clinvar_df_noncoding = pd.read_csv(
    "/s/project/ml4rg_students/2025/project07/group_2/data/clinvar_noncoding_with_preds.csv",
    sep="\t"
)

# 1) Choose device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 2) Re-instantiate your model architecture
model = FineTunedSpeciesLM().to(device)

# 3) Load in the checkpoint
ckpt = torch.load('../checkpoints_regional/best_model_regional.pt', map_location=device)

# 4) Populate your model’s parameters
model.load_state_dict(ckpt)


model.eval()                       

from transformers import AutoTokenizer

# Load tokenizer matching your model
model_name = "johahi/specieslm-metazoa-upstream-k6"  # example, replace with your exact model if needed
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

def encode_batch_specieslm(seqs, proxy_species='homo_sapiens'):
    kmers_stride1 = lambda seq, k=6: [seq[i:i + k] for i in range(len(seq) - k + 1)]
    inputs = [proxy_species + " " + " ".join(kmers_stride1(seq)) for seq in seqs]
    batch_encoding = tokenizer(
        inputs,
        padding=True,
        truncation=True,
        max_length=2003,
        return_tensors="pt"
    )
    return batch_encoding.input_ids.to(device)

BATCH_SIZE = 128  # Try increasing to 512 or 1024 if you have enough GPU memory

preds = []

with torch.no_grad():
    for i in tqdm(range(0, len(clinvar_df_noncoding), BATCH_SIZE)):
        seq_chunk = clinvar_df_noncoding.seq.iloc[i:i+BATCH_SIZE].tolist()
        
        inputs = encode_batch_specieslm(seq_chunk)  # Already moved to device inside
        outputs = model(inputs)
        
        # Move output to CPU only once, outside the GPU
        preds.append(outputs.squeeze().cpu())

preds_tensor = torch.cat(preds)

torch.save(preds_tensor, "clinvar_predictions_regional.pt")















#with torch.no_grad():
#    for i in tqdm(range(0, len(test), BATCH_SIZE)):
#        seq_chunk = test.seq.iloc[i:i+BATCH_SIZE].tolist()
#        inputs = encode_batch_specieslm(seq_chunk)
#        outputs = model(inputs)
#        preds.append(outputs.squeeze().cpu())
#with torch.no_grad():
#    for i in tqdm(range(0, len(clinvar_df_noncoding), BATCH_SIZE)):
#        seq_chunk = clinvar_df_noncoding.seq.iloc[i:i+BATCH_SIZE].tolist()
#        inputs = encode_batch_specieslm(seq_chunk)
#        outputs = model(inputs)
#        preds.append(outputs.squeeze().cpu())
#preds_tensor = torch.cat(preds).cpu()
#torch.save(preds_tensor, "clinvar_predictions.pt")#clinvar_df_noncoding["prediction"] = list(torch.cat(preds).numpy())
#test["prediction"] = torch.cat(preds).numpy()

#clinvar_df_noncoding.to_csv(
# #   "/s/project/ml4rg_students/2025/project07/group_2/data/clinvar_noncoding_with_modelpredictions.tsv",
#    sep="\t",
#    index=False
#)