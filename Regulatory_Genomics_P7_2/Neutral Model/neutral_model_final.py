import pandas as pd
import tqdm
from concurrent.futures import ThreadPoolExecutor
from gnomad_db.database import gnomAD_DB
from pathlib import Path

data_location = '/s/project/benchmark-lm/ssd-cache'
fasta = Path("/s/project/ml4rg_students/2025/project07/data/GRCh38.primary_assembly.genome.fa")
seqs = pd.read_csv("/s/project/ml4rg_students/2025/project07/group_2/data/seqs_2009.csv")

db = gnomAD_DB(data_location, gnomad_version="v4")

all_alleles = {'A', 'C', 'G', 'T'}
columns = ['context', 'ref', 'alt', 'AF']

def prep(chrom, start, end):
    if isinstance(chrom, str) and chrom.lower().startswith("chr"):
        chrom = chrom[3:]
    db_region = db.get_info_for_interval(chrom=chrom, interval_start=start, interval_end=end, query="*")
    db_region = db_region[db_region['filter'] == 'PASS']
    db_region['len_ref'] = db_region['ref'].apply(len)
    db_region['len_alt'] = db_region['alt'].apply(len)
    db_region['is_snv'] = (db_region['len_ref'] == 1) & (db_region['len_alt'] == 1)
    db_region = db_region[db_region['is_snv']]
    return db_region

def get_sevenmer_context(pos, start, sequence):
    central_pos = pos - start
    return sequence[central_pos - 3 : central_pos + 4]

def process_row(row_index_row):
    row_index, row = row_index_row
    local_rows = []
    start = row.Start
    sequence = row.seq

    gnomad_df = prep(row.Chromosome, row.Start + 3, row.End - 3)
    for pos, group in gnomad_df.groupby('pos'):
        seven_mer = get_sevenmer_context(pos, start, sequence)
        if group.empty:
            continue
        ref = group.iloc[0]['ref']
        available_alleles = set()
        for rowalt in group.itertuples(index=False):
            alt = rowalt.alt
            alfreq = rowalt.AF
            available_alleles.add(str(alt))
            local_rows.append([seven_mer, ref, alt, alfreq])
        for allele in all_alleles - available_alleles:
            local_rows.append([seven_mer, ref, allele, 0])
    return local_rows

with ThreadPoolExecutor(max_workers=8) as executor:
    results = list(tqdm.tqdm(executor.map(process_row, seqs.iterrows()), total=len(seqs), desc="Processing rows"))

# Flatten the list of lists into a single list
all_rows = [item for sublist in results for item in sublist]

alfreq_df = pd.DataFrame(all_rows, columns=columns)
alfreq_df.to_csv('data/all_freq.csv', index=False)

