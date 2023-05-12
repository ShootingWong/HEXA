import torch
import torch.nn as nn
import torch.nn.init as init
from graph_embed_speed_nobert_distri import Embed
from transformers import BertModel, BertTokenizer
import pickle
#from graph_embed_speed_newattn_768 import Embed
import copy

# class BCE_Loss_my(nn.Module):
#     def __init__(self):
#         super(BCE_Loss_my, self).__init__()
#         # self.sigmod = torch.nn.Sigmoid()
#     def forward(self, logits, batch_y):
#         y_pred = torch.sigmoid(logits)
#         return -(batch_y * torch.log(y_pred) + (1 - batch_y) * torch.log(1 - y_pred)).mean()

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

def id2str(ids):
    return ' '.join([recab[id_] for id_ in ids])
    
def get_q_d(input_ids):
    EOS = 30522
    raw = copy.deepcopy(input_ids)
    length = len(raw)
    input_ids.reverse()
    ind1 = input_ids.index(EOS)
    input_ids = input_ids[ind1+1:]
    ind2 = input_ids.index(EOS)
    input_ids = input_ids[ind2+1:]
    ind3 = input_ids.index(EOS)
    
    ind3 += ind2+1
    ind2 += ind1+1
    ind3 += ind1+1
    
    ind1 = length-1-ind1
    ind2 = length-1-ind2
    ind3 = length-1-ind3
    
    q_ids = raw[ind3+1: ind2]
    d_ids = raw[ind2+1: ind1]
    
    return q_ids, d_ids

    
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
        
        
class BertRanker(nn.Module):
    def __init__(self, bert_model, config):
        super(BertRanker, self).__init__()
        self.graph_emb_size = config["embsize"]
        self.bert_model = bert_model
        self.classifier = nn.Linear(768, 1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.1)
        init.xavier_normal_(self.classifier.weight)

        #self.graph_ember = Embed(config)

    def forward(self, batch_data):
        """
        Args:
            input_ids ([type]): [description]
            attention_mask ([type]): [description]
            token_type_ids ([type]): [description]
        """
        #sequence modelling

        # print('for batch data')
        # for key in batch_data:
        #     print('BATCH KEY = {}, DATA SIZE = {}'.format(key, batch_data[key].size()))

        input_ids = batch_data["input_ids"]
        attention_mask = batch_data["attention_mask"]
        token_type_ids = batch_data["token_type_ids"]

        bert_inputs = {'input_ids': input_ids, 'attention_mask': attention_mask, 'token_type_ids': token_type_ids}
        sent_rep = self.dropout(self.bert_model(**bert_inputs)[1])

        #graph modelling
        #qid = batch_data["qid"]
        #did = batch_data["did"]

        #batch_size = qid.size(0)
        #node_ids = torch.cat([qid, did], dim=0)

        # print('BEFORE GRAPH')
        # get_gpu()
        #node_embeds = self.graph_ember(node_ids)

        # print('AFTER GRAPH')
        # get_gpu()
        #q_graph_emb = node_embeds[:batch_size]
        #d_graph_emb = node_embeds[batch_size:]

        #graph_rep = q_graph_emb * d_graph_emb  # batch, input_dim

        #classify
        y_pred = self.classifier(sent_rep)

        return y_pred.squeeze(1)