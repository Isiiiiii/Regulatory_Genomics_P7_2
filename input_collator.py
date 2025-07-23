import torch

class SpeciesMaskedCollator:
    def __init__(self, tokenizer, mlm_probability=0.15):
        self.tokenizer = tokenizer
        self.mlm_probability = mlm_probability
        self.mask_token_id = tokenizer.mask_token_id

    def __call__(self, batch):
        input_ids = [item['input_ids'] for item in batch]
        labels = [item['labels'] for item in batch]
        an_arr = [item['an_arr'] for item in batch]
        an_mask = [item['an_mask'] for item in batch]

        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = torch.stack(labels)
        an_arr = torch.stack(an_arr)
        an_mask = torch.stack(an_mask)

        # Clone input_ids for masking
        masked_input_ids = input_ids.clone()

        # Generate mask probabilities
        probability_matrix = torch.full(masked_input_ids.shape, self.mlm_probability)
        special_tokens_mask = input_ids == self.tokenizer.pad_token_id
        probability_matrix.masked_fill_(special_tokens_mask, value=0.0)

        # Apply masks
        masked_indices = torch.bernoulli(probability_matrix).bool()
        masked_input_ids[masked_indices] = self.mask_token_id

        return {
            'input_ids': masked_input_ids,
            'labels': labels,
            'an_arr': an_arr,
            'an_mask': an_mask,
        }
