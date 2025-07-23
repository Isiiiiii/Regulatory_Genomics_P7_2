import numpy as np
import pandas as pd
import torch
import polars as pl
import os

from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader, Subset
from gnomad_db.database import gnomAD_DB
from transformers import AutoTokenizer
from transformers import DataCollatorForLanguageModeling
from tqdm import tqdm

from input_collator import SpeciesMaskedCollator

CACHE_PATH = "/s/project/ml4rg_students/2025/project07/group_2/dataset_cache/precomputed_dataset_cache.pt"


# --- Utility mappings ---
nuc_to_int_dict = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
revcomp_dict = {'A': 'T', 'C': 'G', 'G': 'C', 'T': 'A'}

# --- Utility functions ---
def one_hot_seq(seq):
    seq = seq.upper()
    one_hot = np.zeros((len(seq), 4), dtype=int)
    for i, nucleotide in enumerate(seq):
        if nucleotide in nuc_to_int_dict:
            one_hot[i, nuc_to_int_dict[nucleotide]] = 1
        else:
            raise ValueError(f"Unknown nucleotide {nucleotide} at position {i}")
    return one_hot

def get_af_from_interval(interval, db):
    df_region = db.get_info_for_interval(chrom=interval['Chromosome'].values[0].strip('chr'), 
        interval_start=interval['Start'].values[0] + 1, #IMPORTANT 1 BASED COORDINATES IN GnomAD
        interval_end=interval['End'].values[0], query="*") #END IS INCLUDED

    #only consider SNPs that Pass the gnomad filtering criteria
    df_region = df_region[df_region['filter'] == 'PASS'].copy()
    df_region['len_ref'] = df_region['ref'].apply(len)
    df_region['len_alt'] = df_region['alt'].apply(len)
    df_region = df_region[(df_region['len_ref'] == 1) & (df_region['len_alt'] == 1)].copy().reset_index(drop=True)

    if interval['Strand'].values[0] == '+':
        df_region['relative_pos'] = (df_region['pos'] - (interval['Start'].values[0] + 1)).astype(int)  # this is zero based

        df_region['ref_int'] = df_region['ref'].map(nuc_to_int_dict).astype(int)
        df_region['alt_int'] = df_region['alt'].map(nuc_to_int_dict).astype(int)

        interval_length = interval['End'].values[0] - interval['Start'].values[0]
        afs_arr = np.zeros((interval_length, 4))
        afs_arr[df_region['relative_pos'].values, df_region['alt_int'].values] = df_region['AF'].values

        one_hot_arr = one_hot_seq(interval['seq'].values[0])

        afs_arr[one_hot_arr==1] = 1 - afs_arr.sum(axis=-1)

    elif interval['Strand'].values[0] == '-':
        df_region['relative_pos'] = (interval['End'].values[0] - df_region['pos']).astype(int)

        df_region['ref_int'] = df_region['ref'].map(revcomp_dict).map(nuc_to_int_dict).astype(int)
        df_region['alt_int'] = df_region['alt'].map(revcomp_dict).map(nuc_to_int_dict).astype(int)

        interval_length = interval['End'].values[0] - interval['Start'].values[0]
        afs_arr = np.zeros((interval_length, 4))
        afs_arr[df_region['relative_pos'].values, df_region['alt_int'].values] = df_region['AF'].values

        one_hot_arr = one_hot_seq(interval['seq'].values[0])
        afs_arr[one_hot_arr==1] = 1 - afs_arr.sum(axis=-1)

    return afs_arr, df_region 

def kmers_stride1(seq, k=6):
    return [seq[i:i + k] for i in range(0, len(seq)-k+1)]

def tok_func_species(seq, tokenizer, proxy_species="homo_sapiens", max_length=2003):
    text = "homo_sapiens " + " ".join(kmers_stride1(seq))
    res = tokenizer(
        text,
        truncation=True,
        max_length=max_length,
        return_tensors="pt"
    )
    return res['input_ids'].squeeze(0)



# --- Dataset class ---
class proj_loader(Dataset):
    def __init__(self, precompute, minimum_total_allele_number=150788):
        self.precompute = precompute
        self.minimum_total_allele_number = minimum_total_allele_number
        self.data = []

        # Tokenizer and database
        self.tokenizer = AutoTokenizer.from_pretrained(
            "johahi/specieslm-metazoa-upstream-k6", 
            trust_remote_code=True
        )

        self.db = gnomAD_DB('/s/project/benchmark-lm/ssd-cache', gnomad_version="v4")
        self.scores_lazy = pl.scan_parquet(
            "/s/project/benchmark-lm/data/gnomad4_1_allele_number/"
        )

        # Load sequences
        seqs_df = pd.read_csv('/s/project/ml4rg_students/2025/project07/data/gtf_start_extended_ints_df_2003_seq.csv')
        seqs_df = seqs_df[~seqs_df['seq'].str.contains('N', case=False)]
        self.seqs_df = seqs_df.reset_index(drop=True)

        # Try loading from cache
        if os.path.exists(CACHE_PATH):
            print(f"📦 Loading precomputed dataset from: {CACHE_PATH}")
            self.data = torch.load(CACHE_PATH)
            self.precompute = True
            return

        # If no cache, precompute if requested
        if self.precompute:
            print("⚙️ Precomputing dataset into memory...")
            for idx in tqdm(range(len(self.seqs_df))):
                example = self._process_sample(idx)
                if example is not None:
                    self.data.append(example)
            print(f"💾 Saving precomputed dataset to: {CACHE_PATH}")
            torch.save(self.data, CACHE_PATH)
    
    def __len__(self):
        return len(self.data) if self.precompute else len(self.seqs_df)

    def __getitem__(self, idx):
        if self.precompute:
            return self.data[idx]
        else:
            return self._process_sample(idx)

    def _process_sample(self, idx):
        interval = self.seqs_df.iloc[idx].to_frame().T
        try:
            afs_arr, _ = get_af_from_interval(interval, self.db)
        except Exception as e:
            print(f"⚠️ Failed to fetch interval at index {idx}: {e}")
            return None

        an_df = self.scores_lazy.filter(
            (pl.col("chrom") == interval['Chromosome'].values[0]) & 
            (pl.col("pos") >= interval['Start'].values[0] + 1) & 
            (pl.col("pos") <= interval['End'].values[0])
        ).collect().to_pandas()

        #an_arr = an_df['AN'].values if interval['Strand'].values[0] == '+' else an_df['AN'].values[::-1]
        an_arr = an_df['AN'].values if interval['Strand'].values[0] == '+' else an_df['AN'].values[::-1].copy()
        an_mask = an_arr < self.minimum_total_allele_number

        input_ids = tok_func_species(interval['seq'].values[0], self.tokenizer)

        return {
            'labels': torch.tensor(afs_arr, dtype=torch.float32),
            'an_arr': torch.tensor(an_arr, dtype=torch.long),
            'an_mask': torch.tensor(an_mask, dtype=torch.bool),
            'input_ids': input_ids.long()
        }


# --- DataLoader preparation ---
def prepare_data_loader(batch_size, split_positional:bool, precompute_data=True):
    dataset = proj_loader(precompute_data)
    collator = SpeciesMaskedCollator(
        tokenizer=dataset.tokenizer,
        mlm_probability=0.15
    )
    # if split_positional=True, load the whole dataset as one into dataloader and outputs one dataloader
    if(split_positional):
        data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True, collate_fn=collator)
        return data_loader

    # if split_positional=False -> regional split wanted, splits dataset into 80/20
    elif not split_positional:
        # Step 1: Create per-chromosome split
        train_indices = []
        val_indices = []

        for chrom in dataset.seqs_df['Chromosome'].unique():
            chrom_df = dataset.seqs_df[dataset.seqs_df['Chromosome'] == chrom]
            indices = chrom_df.index.tolist()
            train_idx, val_idx = train_test_split(
                indices, test_size=0.2, random_state=42
            )
            train_indices.extend(train_idx)
            val_indices.extend(val_idx)

        # Step 2: Create subsets and dataloaders
        train_set = Subset(dataset, train_indices)
        val_set = Subset(dataset, val_indices)

        train_loader = DataLoader(
            train_set, batch_size=batch_size, shuffle=True,
            num_workers=8, pin_memory=True, collate_fn=collator
        )
        val_loader = DataLoader(
            val_set, batch_size=batch_size, shuffle=False,
            num_workers=8, pin_memory=True, collate_fn=collator
        )

        return train_loader, val_loader

        




#if __name__ == "__main__":
#    # Example usage:
#    from some_module import gnomAD_DB  # make sure to import your DB class here
#    from transformers import AutoTokenizer#
#
#    seqs_df_path = 'path_to_seqs_df.csv'  # Your local path or URL
#    db_path = '/path/to/gnomad/db'#
#
 #   # Initialize your scores_lazy object here, for example:
  #  scores_lazy = ...  # your lazy polars dataframe
#
 #   # Initialize tokenizer (example with Huggingface tokenizer)
  #  tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
#
 #   # Create DataLoader
  #  data_loader = prepare_data_loader(batch_size=16)
#
 #   # Iterate through DataLoader once (just to test)
  #  for batch in data_loader:
   #     print(batch['input_ids'].shape)
    #    print(batch['labels'].shape)
     #   break
