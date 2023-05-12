import torch
import torch.nn as nn
import torch.nn.init as init
from transformers import BertModel, BertTokenizer
import pickle
import copy


import pynvml
'''
tokenizer = BertTokenizer.from_pretrained('/data/shuting_wang/SubPer/bert')
vocab = tokenizer.vocab
recab = dict()
for key in vocab:
    recab[vocab[key]] = key
print('recab[0] = ', recab[0])
'''

def get_gpu():
    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(0)   #gpu_id
    meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
    print('GPU USED : ', meminfo.used/1024/1024, 'MB')


    
class Ranker_nobert(nn.Module):
    def __init__(self, additional_tokens, config):
        super(Ranker_nobert, self).__init__()
        
        self.dropout = nn.Dropout(0.1)
        self.bert_model_base_q = BertModel.from_pretrained(config['BERT_folder']) 
        self.bert_model_base_d = BertModel.from_pretrained(config['BERT_folder']) 
        self.bert_model_base_q.resize_token_embeddings(self.bert_model_base_q.config.vocab_size + additional_tokens)
        self.bert_model_base_d.resize_token_embeddings(self.bert_model_base_d.config.vocab_size + additional_tokens)

        self.loss = nn.CrossEntropyLoss()

    def forward(self, batch_data, is_test=False):
        """
        Args:
            input_ids ([type]): [description]
            attention_mask ([type]): [description]
            token_type_ids ([type]): [description]
        """

        device = batch_data["input_ids_q"].device

        
        q_input = {'input_ids': batch_data["input_ids_q"], 'attention_mask': batch_data["attention_mask_q"], 'token_type_ids': batch_data["token_type_ids_q"]}
        d_input = {'input_ids': batch_data["input_ids_d"], 'attention_mask': batch_data["attention_mask_d"], 'token_type_ids': batch_data["token_type_ids_d"]}
  

        batch_size = batch_data["input_ids_q"].size(0)
     
        q_emb = self.dropout(self.bert_model_base_q(**q_input)[1])
        d_emb = self.dropout(self.bert_model_base_d(**d_input)[1])
        
        if is_test:
            return (q_emb * d_emb).sum(1)
 
        batch_sim = torch.einsum("ad,bd->ab", q_emb, d_emb)
        batch_label = torch.arange(batch_size).to(device)
        loss = self.loss(batch_sim, batch_label)

        return loss#.squeeze(1)
        
        