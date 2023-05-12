import torch
import torch.nn as nn
import torch.nn.init as init
from graph_emb import Embed

import math

import pynvml

def get_gpu():
    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(0)   #gpu_id
    meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
    print('GPU USED : ', meminfo.used/1024/1024, 'MB')

class Ranker(nn.Module):
    def __init__(self, bert_model, config, heter, weight=None):
        super(Ranker, self).__init__()
        self.graph_emb_size = config["embsize"]
        self.hidden_dim = config['hidden_dim']
        self.bert_model = bert_model
        self.sqrt = math.sqrt(self.graph_emb_size)# * 0.5
        self.classifier = nn.Linear(self.graph_emb_size, 1)
        self.cat = nn.Linear(3, 1, bias=False)
        self.cat.weight.data[:, :] = 1
        
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.1)
        
        self.graph_ember = Embed(config)
        
    def freeze_bert_layer(self, layer):
        trains = ['encoder.layer.{}.'.format(i+9) for i in range(12-layer)]
        for name, param in self.bert_model.named_parameters():
            flag = False
            for skip in trains: 
                if skip in name:
                    flag = True
                    break
            if not flag: continue
            param.requires_grad = False
        for name, param in self.bert_model.named_parameters():
            print(name, param.requires_grad)
    
    def freeze_bert(self):
        for parameter in self.classifier.parameters():
            parameter.requires_grad = False
        for parameter in self.bert_model.parameters():
            parameter.requires_grad = False
        
    def forward(self, input_ids, attention_mask, token_type_ids, qid, did, session_qid, session_did, session_len, label=None):
        """
        Args:
            input_ids ([type]): [description]
            attention_mask ([type]): [description]
            token_type_ids ([type]): [description]
        """

        device = input_ids.device
        batch_size = qid.size(0)
        
        valid_dind = (did != 0)
        valid_did = did[valid_dind]

        bert_inputs = {'input_ids': input_ids, 'attention_mask': attention_mask, 'token_type_ids': token_type_ids}

        sent_rep = self.dropout(self.bert_model(**bert_inputs)[1])
        q_graph_emb, session_intent = self.graph_ember(qid, session_qid, session_did, session_len, batch_size)
        d_emb = torch.Tensor(self.graph_ember.content_encoder.id2emb[did.detach().cpu().numpy()]).to(device)

        graph_rep = q_graph_emb * d_emb
        session_rep = session_intent * d_emb
        
        y_pred = self.cat(torch.cat([graph_rep.sum(1, keepdim=True) / self.sqrt, session_rep.sum(1, keepdim=True) / self.sqrt, self.classifier(sent_rep)], dim=1)) 

        return y_pred.squeeze(1)
        
    def test(self, input_ids, attention_mask, token_type_ids, qid, did, session_qid, session_did, session_len, label=None):

        device = input_ids.device
        batch_size = input_ids.size(0)
        q_graph_emb, session_intent = self.graph_ember(qid, session_qid, session_did, session_len, batch_size)
                                                                    
        y_pred_all = []
        candi_cnt = did.size(1)
        
        for i in range(candi_cnt):
            input_ids_per = input_ids[:, i, :]
            attention_mask_per = attention_mask[:, i, :]
            token_type_ids_per = token_type_ids[:, i, :]
            did_per = did[:, i]
            bert_inputs = {'input_ids': input_ids_per, 'attention_mask': attention_mask_per, 'token_type_ids': token_type_ids_per}
            sent_rep = self.dropout(self.bert_model(**bert_inputs)[1])
            
            d_emb = torch.Tensor(self.graph_ember.content_encoder.id2emb[did_per.detach().cpu().numpy()]).to(device)
            graph_rep = q_graph_emb * d_emb
            session_rep = session_intent * d_emb

            y_pred = self.cat(torch.cat(
                [graph_rep.sum(1, keepdim=True) / self.sqrt, session_rep.sum(1, keepdim=True) / self.sqrt, self.classifier(sent_rep)], dim=1))
            
            y_pred_all.append(y_pred)
        y_pred_all = torch.cat(y_pred_all, dim=1)


        return y_pred_all
      
class BertRanker(nn.Module):
    def __init__(self, bert_model, config):
        super(BertRanker, self).__init__()
        self.graph_emb_size = config["embsize"]
        self.bert_model = bert_model
        self.classifier = nn.Linear(768, 1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.1)
        init.xavier_normal_(self.classifier.weight)

    def freeze_bert_layer(self, layer):
        trains = ['encoder.layer.{}.'.format(i+9) for i in range(12-layer)]
        for name, param in self.bert_model.named_parameters():
            flag = False
            for skip in trains: 
                if skip in name:
                    flag = True
                    break
            if not flag: continue
            param.requires_grad = False
        for name, param in self.bert_model.named_parameters():
            print(name, param.requires_grad)

    def forward(self, input_ids, attention_mask, token_type_ids, qid, did, session_qid, session_did, session_len, label=None):
        """
        Args:
            input_ids ([type]): [description]
            attention_mask ([type]): [description]
            token_type_ids ([type]): [description]
        """

        bert_inputs = {'input_ids': input_ids, 'attention_mask': attention_mask, 'token_type_ids': token_type_ids}
        sent_rep = self.dropout(self.bert_model(**bert_inputs)[1])

        #classify
        y_pred = self.classifier(sent_rep)

        return y_pred.squeeze(1)
        
    def test(self, input_ids, attention_mask, token_type_ids, qid, did, session_qid, session_did, session_len, label=None):

        device = input_ids.device
        batch_size = input_ids.size(0)
        y_pred_all = []
        candi_cnt = did.size(1)
        for i in range(candi_cnt):
            input_ids_per = input_ids[:, i, :]
            attention_mask_per = attention_mask[:, i, :]
            token_type_ids_per = token_type_ids[:, i, :]
            did_per = did[:, i]
            bert_inputs = {'input_ids': input_ids_per, 'attention_mask': attention_mask_per, 'token_type_ids': token_type_ids_per}
            sent_rep = self.dropout(self.bert_model(**bert_inputs)[1])

            y_pred = self.classifier(sent_rep)
            y_pred_all.append(y_pred)
        y_pred_all = torch.cat(y_pred_all, dim=1)

        return y_pred_all
        

class Ranker_emb(nn.Module):
    def __init__(self, config, weight=None):
        super(Ranker_emb, self).__init__()
        self.graph_emb_size = config["embsize"]
        self.hidden_dim = config['hidden_dim']
        self.graph_ember = Embed(config)
        
    def forward(self, input_ids, attention_mask, token_type_ids, qid, did, session_qid, session_did, session_len, label=None):
        """
        Args:
            input_ids ([type]): [description]
            attention_mask ([type]): [description]
            token_type_ids ([type]): [description]
        """
        
        d_emb = torch.Tensor(self.graph_ember.content_encoder.id2emb[did.detach().cpu().numpy()]).cuda()
        q_emb = torch.Tensor(self.graph_ember.content_encoder.id2emb[qid.detach().cpu().numpy()]).cuda()
        
        ad_rep = q_emb * d_emb
        
        y_pred = ad_rep.sum(1)
        
        return y_pred
        
    def test(self, input_ids, attention_mask, token_type_ids, qid, did, session_qid, session_did, session_len, label=None):

        q_emb = torch.Tensor(self.graph_ember.content_encoder.id2emb[qid.detach().cpu().numpy()]).cuda()
        
        device = input_ids.device
        batch_size = input_ids.size(0)
        y_pred_all = []
        candi_cnt = did.size(1)
        for i in range(candi_cnt):
            d_emb = torch.Tensor(self.graph_ember.content_encoder.id2emb[did[:,i].detach().cpu().numpy()]).cuda()
            
            ad_rep = q_emb * d_emb
        
            y_pred_1 = ad_rep.sum(1, keepdim=True)
            y_pred_all.append(y_pred_1)
        y_pred_all = torch.cat(y_pred_all, dim=1)

        return y_pred_all
        