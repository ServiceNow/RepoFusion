import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from transformers import AutoModel


class RuleModel(nn.Module):
    def __init__(self, emb_model_type, repr_size=768, device='cpu', n_head=1, d_k=128, d_proj=512,
                 mode='rlpg-h', dropout=0.0):
        super(RuleModel, self).__init__()

        self.n_rules = 63
        self.set_embedding_model(emb_model_type)
        self.repr_size = repr_size
        self.device = device
        self.mode = mode

        if self.mode == 'rlpg-h':
          self.hole_dense1 = nn.Linear(self.repr_size, d_proj)
          self.hole_dense2 = nn.Linear(d_proj, self.n_rules)
        

    def get_representation(self, inputs, mask):
        outputs = self.emb_model(inputs, attention_mask=mask)
        try:
            representation = outputs.pooler_output
        except:
            representation = outputs.last_hidden_state[:, 0]
        return representation

    def get_context_embedding(self, context, attn_mask):
        context_embedding = self.get_representation(context, attn_mask)
        return context_embedding

    def forward(self, info):

        hole_inputs, hole_mask = info
        hole_window_repr = self.get_context_embedding(hole_inputs, hole_mask)
        #get prediction from hole window
        if self.mode == 'rlpg-h':           
            hole_pred = self.hole_dense2(F.relu(self.hole_dense1(hole_window_repr)))
            if len(hole_pred.shape)==1:
              hole_pred = torch.unsqueeze(hole_pred, dim=0)
            hole_pred = torch.sigmoid(hole_pred)
            return hole_pred

    def set_embedding_model(self, emb_model_type):
        # CodeBERT
        if emb_model_type == 'codebert':
          self.emb_model = AutoModel.from_pretrained("microsoft/codebert-base")

        # freeze the parameters of the pretrained emb_model
        for param in self.emb_model.parameters():
            param.requires_grad = False