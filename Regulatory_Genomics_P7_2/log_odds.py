import torch.nn.functional as F
import torch
import numpy as np
from tqdm import tqdm

def nuc_pos_index(nucleotide):
    if nucleotide == 'A':
        one_hot = 0
    elif nucleotide == 'C':
        one_hot = 1
    elif nucleotide == 'G':
        one_hot = 2
    elif nucleotide == 'T':
        one_hot = 3
    else:
        raise ValueError(f"Unknown nucleotide {nucleotide} at position {i} in sequence {seq}")
    return one_hot

def log_odds_correction(model_df, neutral_df, predictions, rel_pos=False):
    '''Gives the difference between the log(odds) for the polymorphism at a position from 
    the predictions of our model and the log(odds) for the polymorphism at a position in the
    neutral model. 
    inputs:
    - model_df: dataframe that has columns Seq, Rel_pos, Ref
    - neutral_df: neutral model dataframe
    - predictions: tensor of probabilities and shape [B, 2003, 4]
    - rel_pos: True/False, relative position of nucleotide in the sequence is known
    outputs:
    - alt_probs_sumes_list: list of all probabilities for polymorphism at a position
    - log_odds_list: list of computed log odds differences'''
    

    alt_probs_sumed_list = []
    log_odds_list = []
    not_matched = 0

    for idx, row in model_df.iterrows():

        sample_number = idx

        seq = row['Seq']
        ref = row['Ref']

        if rel_pos:
            pos = row['Rel_pos']
        else:
            pos = 1002

        # Get the predictions for our sample
        probs_sample = predictions[sample_number] #(2003, 4)

        # Get the prediction for our position in sequence
        probs_position = probs_sample[pos]

        # If the refrences align
        if ref == seq[pos]:

            # Positions for the 7-mer context
            start = pos - 3
            end = pos + 4
            seven_mer = seq[start:end]

            # Locate in the neutral model
            neutral_rows = neutral_df[neutral_df['sequence_context'] == seven_mer]

            # Sum of alternative probs - neutral model
            # TODO: change to correct value
            neutral_probs_sumed = neutral_rows['calibrated_cluster_reg'].sum()

            # Remove value for ref nucleotide in model predictions
            ref_index = nuc_pos_index(ref)
            mask = torch.arange(4) != ref_index
            probs_reduced = probs_position[mask]

            # Sum the alternative result
            model_probs_sumed = probs_reduced.sum()
            alt_probs_sumed_list.append(model_probs_sumed)

            # Odds
            epsilon = 1e-8
            odds_predicted = (model_probs_sumed + epsilon) / (1-model_probs_sumed + epsilon)
            odds_neutral = (neutral_probs_sumed + epsilon) / (1- neutral_probs_sumed + epsilon)

            # Optional: clip odds to avoid log(0) or log(very large)
            odds_predicted = np.clip(odds_predicted, epsilon, 1e8)
            odds_neutral = np.clip(odds_neutral, epsilon, 1e8)

            # Safe log-odds
            log_odds_final = np.log(odds_predicted) - np.log(odds_neutral)
            log_odds_list.append(log_odds_final.item())

        else:
            not_matched += 1

    print(f'Not matched: {not_matched}')

    return alt_probs_sumed_list, log_odds_list


def log_odds_correction_logits(model_df, neutral_df, predictions_logits, rel_pos=None):
    '''Gives the difference between the log(odds) for the polymorphism at a position from 
    the predictions of our model and the log(odds) for the polymorphism at a position in the
    neutral model. To avoid computational problems it expect the predictions in logits.
    inputs:
    - model_df: dataframe that has columns Seq, Rel_pos, Ref
    - neutral_df: neutral model dataframe
    - predictions_logits: tensor of logits and shape [B, 2003, 4]
    - rel_pos: if None take from df
    outputs:
    - log_odds_list: list of computed log odds differences'''
    

    log_odds_list = []
    not_matched = 0

    
    for idx, row in tqdm(model_df.iterrows(), total=len(model_df), desc="Computing log-odds"):

        sample_number = idx

        seq = row['Seq']
        ref = row['Ref']

        if rel_pos is None:
            pos = row['Rel_pos']
        else:
            pos = rel_pos

        # Get the predictions for our sample
        logits_sample = predictions_logits[sample_number] #(2003, 4)

        # Get the prediction for our position in sequence
        logits_position = logits_sample[pos] #(4)

        # If the refrences align
        if ref == seq[pos]:

            # Positions for the 7-mer context
            start = pos - 3
            end = pos + 4
            seven_mer = seq[start:end]

            # Locate in the neutral model
            neutral_rows = neutral_df[neutral_df['sequence_context'] == seven_mer]

            # Sum of alternative probs - neutral model
            # TODO: change to correct value
            neutral_probs_sumed = neutral_rows['calibrated_cluster_reg'].sum()

            # Remove value for ref nucleotide in model predictions
            ref_index = nuc_pos_index(ref)
            mask = torch.arange(4) != ref_index
            logits_alt = logits_position[mask]

            # log(sum_{ALT} exp(logit_i))
            logsumexp_alt = torch.logsumexp(logits_alt, dim=0)

            # Log(odds) of predicted
            log_odds_predicted = logsumexp_alt - logits_position[ref_index]

            # Odds of neutral
            epsilon = 1e-8
            odds_neutral = (neutral_probs_sumed + epsilon) / (1- neutral_probs_sumed + epsilon)

            # Optional: clip odds to avoid log(0) or log(very large)
            odds_neutral = np.clip(odds_neutral, epsilon, 1e8)

            # Log(odds) neutral
            log_odds_neutral = np.log(odds_neutral)
            log_odds_final = log_odds_predicted - log_odds_neutral
            log_odds_list.append(log_odds_final.item())

        else:
            not_matched += 1

    print(f'Not matched: {not_matched}')

    return log_odds_list