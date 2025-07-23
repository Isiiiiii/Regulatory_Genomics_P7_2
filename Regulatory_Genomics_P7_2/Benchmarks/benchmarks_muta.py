import torch
from model_architecture import FineTunedSpeciesLM
import numpy as np
import pandas as pd
from tqdm.auto import tqdm



kircher_path='/s/project/ml4rg_students/2025/project07/data/kircher_all_metrics_w_paper_w_gpn_msa_w_embs.tsv'

kircher_df = pd.read_csv(kircher_path, sep="\t")

# 1) Choose device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 2) Re-instantiate your model architecture
model = FineTunedSpeciesLM().to(device)

# 3) Load in the checkpoint
ckpt = torch.load('checkpoints_positional/best_model.pt', map_location=device)

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

BATCH_SIZE = 256
preds = []

#with torch.no_grad():
#    for i in tqdm(range(0, len(test), BATCH_SIZE)):
#        seq_chunk = test.seq.iloc[i:i+BATCH_SIZE].tolist()
#        inputs = encode_batch_specieslm(seq_chunk)
#        outputs = model(inputs)
#        preds.append(outputs.squeeze().cpu())
with torch.no_grad():
    for i in tqdm(range(0, len(kircher_df), BATCH_SIZE)):
        seq_chunk = kircher_df.five_prime_seq.iloc[i:i+BATCH_SIZE].tolist()
        inputs = encode_batch_specieslm(seq_chunk)
        outputs = model(inputs)
        
        preds.append(outputs.squeeze().cpu())

preds_tensor = torch.cat(preds).cpu()
torch.save(preds_tensor, "kircher_predictions_positional.pt")

#kircher_df["prediction"] = list(torch.cat(preds))
#test["prediction"] = torch.cat(preds).numpy()

#kircher_df.to_csv(
#    "/s/project/ml4rg_students/2025/project07/group_2/data/kircher_df_with_modelpredictions.tsv",
#    sep="\t",
#    index=False
#)