import pandas as pd
from gnomad_db.database import gnomAD_DB
from pathlib import Path
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
import tqdm
from multiprocessing import Pool
import torch 
import pickle


data_location = '/s/project/benchmark-lm/ssd-cache'
fasta = Path("/s/project/ml4rg_students/2025/project07/data/GRCh38.primary_assembly.genome.fa")
seqs = pd.read_csv("/s/project/ml4rg_students/2025/project07/group_2/data/seqs_2009.csv")

db = gnomAD_DB(data_location, gnomad_version="v4")

def prep(chrom, start, end):

    if isinstance(chrom, str) and chrom.lower().startswith("chr"):
        chrom = chrom[3:]
    
    db_region = db.get_info_for_interval(chrom=chrom, interval_start=start, interval_end = end, query="*")
    db_region = db_region[db_region['filter'] == 'PASS'] 

    interval_length = end - start
    db_region['len_ref'] = db_region['ref'].apply(len)
    db_region['len_alt'] = db_region['alt'].apply(len)  
    db_region['is_snv'] = (db_region['len_ref'] == 1) & (db_region['len_alt'] == 1)

    db_region = db_region[db_region['is_snv']]

    return db_region

def extract_kmers(seq, k=7):
    return [seq[i:i+k] for i in range(len(seq) - k + 1)]

def process_row(row):
    region = prep(chrom=row.Chromosome, start=row.Start, end=row.End)
    region['zero_pos'] = region['pos'] - row.Start - 1
    
    all_kmers = extract_kmers(row.seq, k=7)
    total_kmer_counts = Counter(all_kmers)

    variation_kmer_counts = Counter()
    variation_kmer_alleles = defaultdict(list)

    valid = region[(region['zero_pos'] >= 3) & (region['zero_pos'] <= len(row.seq) - 4)]
    
    #for zero_pos in valid['zero_pos']:
    #    kmer = row.seq[zero_pos - 3 : zero_pos + 4]
    #    variation_kmer_counts[kmer] += 1
    #    variation_kmer_alleles[kmer].append((var['ref'], var['alt']))
    for _, var in valid.iterrows():
        zero_pos = var['zero_pos']
        kmer = row.seq[zero_pos - 3 : zero_pos + 4]
        variation_kmer_counts[kmer] += 1
        variation_kmer_alleles[kmer].append((var['ref'], var['alt']))

    print('allele')    

    return total_kmer_counts, variation_kmer_counts, variation_kmer_alleles

output_file = "partial_results.pkl"

# If you want to resume, load existing results
try:
    with open(output_file, "rb") as f:
        results = pickle.load(f)
        start_idx = len(results)
        print(f"Resuming from row {start_idx}")
except (FileNotFoundError, EOFError):
    results = []
    start_idx = 0
    print("Starting from scratch")

# Process and save one by one
for i, row in enumerate(seqs.itertuples(index=False), 0):
    if i < start_idx:
        continue  # skip already-processed rows

    try:
        res = process_row(row)
        results.append(res)

        # Save to pickle after each processed row
        with open(output_file, "wb") as f:
            pickle.dump(results, f)

        print(f"Processed and saved row {i + 1}")

    except Exception as e:
        print(f"Error at row {i + 1}: {e}")
        break
