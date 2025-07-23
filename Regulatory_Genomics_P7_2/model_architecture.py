import torch
import torch.nn as nn
from transformers import AutoModelForMaskedLM

class FineTunedSpeciesLM(nn.Module):
    def __init__(self, model_name= "johahi/specieslm-metazoa-upstream-k6", seq_len=2003, base_classes=4):
        super().__init__()
        
        base_model = AutoModelForMaskedLM.from_pretrained(model_name, trust_remote_code=True)
        
        self.model_name = model_name
        # Extract embeddings and encoder
        self.embeddings = base_model.bert.embeddings
        self.encoder = base_model.bert.encoder

        #freeze the encoder part
        for param in self.embeddings.parameters():
            param.requires_grad = False
        for param in self.encoder.parameters():
            param.requires_grad = False

        # ConvTranspose head for upsampling from 1998 → 2003
        self.head = nn.ConvTranspose1d(
            in_channels=base_model.config.hidden_size,  # 768
            out_channels=base_classes,            # 4
            kernel_size=6,
            stride=1,
            padding=0
        )
        
        
    def forward(self, input_ids, attention_mask=None, token_type_ids=None):

        #print(" INPUT SIZE:", input_ids.shape)
        x = self.embeddings(
            input_ids=input_ids,
            token_type_ids=token_type_ids
        )
        
        #print(" AFTER EMBEDDINGS: ", x.shape)  
        x = self.encoder(x, attention_mask=attention_mask)[0]
        #print(" AFTER ENCODER: ", x.shape)

        # x: [B, 2001, 768] → remove special tokens
        x = x[:, 2:-1, :]             # [B, 1998, 768]
        #print(" AFTER REMOVING SPECIAL TOKENS: ", x.shape )
        x = x.permute(0, 2, 1)        # [B, 768, 1998]
        #print(" AFTER PERMUTATION: ", x.shape)
        x = self.head(x)             # [B, 4, 2003]
        #print(" AFTER TRANSPOSE CONV HEAD: ", x.shape)
        x = x.permute(0, 2, 1)        # [B, 2003, 4]
        #print(" AFTER PERMUTATION: ", x.shape)

        return x









