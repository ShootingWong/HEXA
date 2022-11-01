'''
Difference with graph_embeded_speed: in this code, the attention is 2-layer mlp on concate with LeakyReLU as activation
edge atten is in the same way.
'''
import numpy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.autograd import Variable
from transformers import BertTokenizer, BertModel
#from torch_geometric.nn.inits import glorot, uniform
from utils import *
import pickle
# from transformer import *
import numpy as np
from time import time
import math

import numpy


# Based on a fixed neighborhood dict

class Embed(nn.Module):
    def __init__(self, config):
        super(Embed, self).__init__()

        self.embsize = config['embsize']
        self.edge_num = config['edge_num']
        #self.split = config['split_cnt']
        self.hidden_dim = config['hidden_dim']
        self.max_session = config['max_session']

        n_heads = config['n_heads']
        num_types = config['num_types']
        use_norm = bool(config['use_norm'])
        self.d_k = self.embsize // n_heads
        self.loc = -1
        self.graph_path = config['sample_neighbor_prefix']

        self.nodetype_dict = pickle.load(open(config['id2type_addr'], 'rb'))
        self.load_graph_info()

        self.content_encoder = BERT_encoder(config)

        neig_sizes = [5, 1, 10, 12, 14]

        self.encoder1 = Aggregtor_HGT(self.content_encoder, config, self.nodetype_dict, self.sample_prefix,
                                   self.embsize, n_heads, num_types, self.edge_num, neig_sizes[0],
                                   config['graph_drt'], use_norm)  # 8
        self.encoder2 = Aggregtor_HGT(lambda nodes, step: self.encoder1(nodes, step), config, self.nodetype_dict, self.sample_prefix,
                                   self.embsize, n_heads, num_types, self.edge_num, neig_sizes[0],
                                   config['graph_drt'], use_norm)  # 4

        self.encoder = self.encoder2  # self.encoder4
        
        #self.gate = nn.Sequential(nn.Linear(self.embsize * 2, self.embsize), nn.Tanh())

    def load_graph_info(self):
        self.sample_prefix = [] #* self.split
        
        #print('LOAD GRAPH WITH LOC = ', loc)
        '''
        for split in range(self.split):
            for e in range(self.edge_num):
                self.sample_prefix[split].append(
                    np.expand_dims(pickle.load(open(config['sample_neighbor_prefix'].format(e, split), 'rb')), axis=0))
            self.sample_prefix[split] = np.expand_dims(np.concatenate(self.sample_prefix[split]), axis=0)
        self.sample_prefix = np.concatenate(self.sample_prefix) # split_cnt, edge_cnt, node_cnt, 100
        '''
        for e in range(self.edge_num):
            self.sample_prefix.append(
                np.expand_dims(pickle.load(open(self.graph_path.format(e), 'rb')), axis=0))
        self.sample_prefix = np.concatenate(self.sample_prefix)

        print('sample fix shape = {} nodetyp len = {}'.format(
            self.sample_prefix.shape, len(self.nodetype_dict)
        ))

    def forward(self, node_ids, session_qids, session_dids, session_len, qcnt=None):
        '''
        :param
        node_ids: the node need to sample the cross-session sub-graph; [batchsize]
        session_qids: the session(q) need to calculate the intra-session message passing; [batch, session_len, ]
        session_dids: the session(d) need to calculate the intra-session message passing; [batch, session_len, ]
        session_len: the actual length of each session in the batch; [batch,]
        :return:
        '''

        batch_size = int(node_ids.size(0) / 2)

        query_emb = self.encoder(node_ids, 1)
        '''
        all_emb = self.encoder(node_ids)
        
        if qcnt:
            query_emb = all_emb[: qcnt, :]
            doc_emb = all_emb[qcnt:, :]
        else:
            query_emb = all_emb[: batch_size, :]
            doc_emb = all_emb[batch_size:, :]
        '''
        
        # raw_query = self.encoder1.features(node_ids)[:batch_size]

        session_intent = self.encoder1.session_graph(query_emb, session_qids, session_dids,
                                                     session_len)  # batch, hidden_size
        session_intent[session_len == 0] = query_emb[session_len == 0]

        # z = self.gate(torch.cat([query_emb, session_intent], dim=1))
        # query_emb = z*query_emb + (1-z)*session_intent

        return query_emb, session_intent  # node_num * emb_dim


class Aggregtor_HGT(nn.Module):
    def __init__(self, features, config, nodetype_dict, sample_prefix,
                 input_dim, n_heads=6, num_types=2, num_relations=8, sample_size=-1, dropout=0.2, use_norm=True):
        super(Aggregtor_HGT, self).__init__()
        
        self.embsize = config['embsize']
        self.edge_num = config['edge_num']
        #self.split = config['split_cnt']
        self.hidden_dim = config['hidden_dim']
        self.max_session = config['max_session']

        n_heads = config['n_heads']
        num_types = config['num_types']
        use_norm = bool(config['use_norm'])
        self.d_k = self.embsize // n_heads
        self.loc = -1
        self.graph_path = config['sample_neighbor_prefix']
        
        self.nodetype_dict = nodetype_dict
        self.sample_prefix = sample_prefix
        self.sample_size = sample_size
        self.edge_num = num_relations
        self.num_types = num_types

        self.input_dim = input_dim
        self.num_types = num_types
        self.num_relations = num_relations
        self.total_rel = num_types * num_relations * num_types
        self.n_heads = n_heads
        self.d_k = input_dim // n_heads
        self.sqrt_dk = math.sqrt(self.d_k)
        self.use_norm = use_norm
        self.att = None

        self.features = features
        
        self.k_linears = nn.Linear(self.embsize, 2 * self.embsize)
        self.q_linears = nn.Linear(self.embsize, 2 * self.embsize)
        self.v_linears = nn.Linear(self.embsize, 2 * self.embsize)
        self.a_linears = nn.Linear(self.embsize, 2 * self.embsize)

        self.norms = nn.ModuleList()
        if use_norm:
            for i in range(num_types):
                self.norms.append(nn.LayerNorm(self.embsize))

        self.relation_pri = nn.Parameter(torch.ones(self.edge_num, n_heads))
        self.relation_att = nn.Parameter(torch.Tensor(self.edge_num, n_heads, self.d_k, self.d_k))
        self.relation_msg = nn.Parameter(torch.Tensor(self.edge_num, n_heads, self.d_k, self.d_k))
        self.skip = nn.Parameter(torch.ones(num_types+1))
        
        self.pos_embedding = nn.Embedding(self.max_session, self.embsize)
        
        self.sess_mlp = nn.Linear(self.embsize * 2, self.embsize)
        self.sess_w = nn.Linear(self.embsize * 2, self.embsize)
        self.sess_a = nn.Linear(self.embsize, 1, bias=False)
        
        self.mid_linear  = nn.Linear(self.embsize,  self.embsize * 2)
        self.out_linear  = nn.Linear(self.embsize * 2,  self.embsize)
        self.out_norm    = nn.LayerNorm(self.embsize)
        
        #glorot(self.relation_att)
        #glorot(self.relation_msg)
        torch.nn.init.xavier_uniform_(self.relation_att)
        torch.nn.init.xavier_uniform_(self.relation_msg)
        
        self.drop = nn.Dropout(dropout)

        self.q_edge = [0, 2, 4, 5]
        self.d_edge = [1, 3, 6, 7]

    def exclusive_combine3(self, neigh_lists):
        # neigh_lists : [edge_num, node, neighbors]
        # cut_nei_list = [[[]]*neigh_lists.size(1)] * neigh_lists.size(0)
        node2ind = dict()
        flat_nodes = set(sum(sum(neigh_lists, []), []))
        # print('flat nodes = ', flat_nodes)
        if 0 in flat_nodes:
            flat_nodes.remove(0)
        flat_nodes = list(flat_nodes)
        for i, node in enumerate(flat_nodes):
            node2ind[node] = i
            # flat_nodes[i] = node[0]

        return flat_nodes, node2ind  # , len(flat_nodes_type) - doc_cnt

    def exclusive_combine2(self, neigh_lists):
        # neigh_lists : [dim1, dim2]
        node2ind = dict()
        flat_nodes = set(sum(neigh_lists, []))
        # print('flat nodes = ', flat_nodes)
        if 0 in flat_nodes:
            flat_nodes.remove(0)
        flat_nodes = list(flat_nodes)
        for i, node in enumerate(flat_nodes):
            node2ind[node] = i
            # flat_nodes[i] = node[0]

        return flat_nodes, node2ind  # , len(flat_nodes_type) - doc_cnt

    def sample_neighs_all(self, nodes, nodes_numpy, size):
        #loc:batch; sample_prefix: split_cnt, edge_cnt, node_cnt, 100 -> split_cnt, edge_cnt, batch, 100
        #sample_prefix = self.sample_prefix[loc]

        rand_num = np.random.random_integers(99, size=(size))
        device = nodes.device
        node_prefix = torch.LongTensor(self.sample_prefix[:, nodes_numpy, :]).to(device)#edge_cnt, batch, 100

        select_neighbor = node_prefix[:, :, rand_num]  # 8, batch, size
        add_index = select_neighbor.sum(-1) == 0  # 8, batch
        select_neighbor[add_index] = nodes.unsqueeze(0).repeat([self.edge_num, 1])[add_index].unsqueeze(-1)

        return select_neighbor

    def get_type_emb(self, node_embs, types, qkv):
        device = node_embs.device

        if qkv == 'q':
            hiddens = self.q_linears(node_embs)  # batch, 2*hiddensize
        elif qkv == 'k':
            hiddens = self.k_linears(node_embs)  # batch, 2*hiddensize
        elif qkv == 'v':
            hiddens = self.v_linears(node_embs)  # batch, 2*hiddensize
        else:
            hiddens = self.a_linears(node_embs)  # batch, 2*hiddensize
        hidden_size = int(hiddens.size(1) / 2)
        hiddens = hiddens.view([node_embs.size(0), 2, hidden_size])
        mask = torch.zeros(node_embs.size(0), 2).to(device).scatter_(1,
                                                                     torch.LongTensor(types).unsqueeze(-1).to(device),
                                                                     1)
        hiddens = (hiddens * mask.unsqueeze(-1)).sum(1)

        return hiddens

    def deal_same_node_session(self, centers):
        #centers: batch, session_len
        same = (centers.unsqueeze(2) == centers.unsqueeze(1))#batch, session_len, session_len
        #mask the left same node, leave one
        repeat = torch.triu(same, 1).sum(1).bool()#uptriple: batch, session_len
        return same, repeat

        
    def session_graph_message(self, center_ids, centers, all_embs, pre_mask, back_mask, click_mask, blank, batch, max_sess_len, type=0):
        #center_ids: batch, session_len | centers: batch, session_len, 768
        big_batch = batch * max_sess_len
        q = self.get_type_emb(centers, [type] * (big_batch), qkv='q').view(batch, max_sess_len, self.n_heads,
                                                                                    self.d_k).transpose(0, 2)\
                                                                                    .transpose(1, 2).contiguous().view(self.n_heads*batch, max_sess_len, self.d_k)#n_head* batch, session_len, d_k
        k = self.get_type_emb(all_embs,
                              [0] * (big_batch) + [1] * (big_batch),
                              qkv='k').view(2, batch, max_sess_len, self.n_heads, self.d_k)\
                              .transpose(0, 3).transpose(2, 3).contiguous().view(self.n_heads*batch, 2*max_sess_len, self.d_k)#n_head* batch, 2* session_len, d_k
        v = self.get_type_emb(all_embs,
                              [0] * (big_batch) + [1] * (big_batch),
                              qkv='v').view(2, batch, max_sess_len, self.n_heads, self.d_k)\
                              .transpose(0, 3).transpose(2, 3).contiguous().view(self.n_heads*batch, 2*max_sess_len, self.d_k)#n_head* batch, 2* session_len, d_k

        useful_edge = [1, 6, 7] if type == 1 else [0, 4, 5]#d_click, d_back, d_pre | q_click, q_back, q_pre
        edge_num = len(useful_edge)
        unmaksed_atten = []
        pass_message = []
        masks = []

        for i, edge in enumerate(useful_edge):
            #relation_att: edge_num, n_heads, d_k, d_k
            k_proj = torch.bmm(k, self.relation_att[edge].unsqueeze(1).repeat([1,batch,1,1]).view(self.n_heads*batch, self.d_k, self.d_k))  # n_head* batch, 2* session_len, dk
            # self.n_heads, batch, all_cnt
            # atten = q * W_e * k
            # relation_pri: edge_num, heads
            atten_i = torch.bmm(q, k_proj.transpose(1, 2).contiguous()) * self.relation_pri[edge].unsqueeze(-1).repeat([1, batch]).view(-1).unsqueeze(-1).unsqueeze(
                -1) / self.sqrt_dk # n_head* batch, session_len, 2* session_len
            unmaksed_atten.append(atten_i.unsqueeze(-1))  # n_head* batch, session_len, 2* session_len, 1
            pass_message.append(torch.bmm(v, self.relation_msg[edge].unsqueeze(1).repeat([1,batch,1,1]).view(self.n_heads*batch, self.d_k, self.d_k)).unsqueeze(2))  # n_head* batch, 2* session_len, 1,  d_k
            if i == 0:
                #d click
                if type == 1:
                    masks.append(torch.cat([click_mask, blank], dim=-1).unsqueeze(0).repeat([self.n_heads, 1, 1, 1]).view(self.n_heads*batch, max_sess_len, 2*max_sess_len, 1))
                else:
                    masks.append(torch.cat([blank, click_mask], dim=-1).unsqueeze(0).repeat([self.n_heads, 1, 1, 1]).view(self.n_heads*batch, max_sess_len, 2*max_sess_len, 1))
            elif i == 1:
                if type == 1:
                    masks.append(torch.cat([blank, back_mask], dim=-1).unsqueeze(0).repeat([self.n_heads, 1, 1, 1]).view(self.n_heads*batch, max_sess_len, 2*max_sess_len, 1))
                else:
                    masks.append(torch.cat([back_mask, blank], dim=-1).unsqueeze(0).repeat([self.n_heads, 1, 1, 1]).view(self.n_heads*batch, max_sess_len, 2*max_sess_len, 1))
            else:
                if type == 1:
                    masks.append(torch.cat([blank, pre_mask], dim=-1).unsqueeze(0).repeat([self.n_heads, 1, 1, 1]).view(self.n_heads*batch, max_sess_len, 2*max_sess_len, 1))
                else:
                    masks.append(torch.cat([pre_mask, blank], dim=-1).unsqueeze(0).repeat([self.n_heads, 1, 1, 1]).view(self.n_heads*batch, max_sess_len, 2*max_sess_len, 1))
             
        pass_message = torch.cat(pass_message, dim=2).view(self.n_heads*batch, 2*max_sess_len * edge_num,
                                                           self.d_k)  # n_head* batch, 2*max_sess_len*3, dk
        unmaksed_atten = torch.cat(unmaksed_atten, dim=3).view(self.n_heads*batch, max_sess_len,
                                                               2*max_sess_len * edge_num)  #  n_head* batch, max_sess_len, 2*max_sess_len*3

        mask = torch.cat(masks, dim=3).view(self.n_heads*batch, max_sess_len,
                                                               2*max_sess_len * edge_num).bool()
        same_, repeat_ = self.deal_same_node_session(center_ids)# same: batch, sesslen, sesslen, bool | repeat: batch, sess_len, bool
        same = (same_.unsqueeze(0).repeat([self.n_heads, 1, 1, 1])).view(self.n_heads*batch, max_sess_len, max_sess_len)
        repeat = (repeat_.unsqueeze(0).repeat([self.n_heads, 1, 1])).view(self.n_heads*batch, max_sess_len)
        for i in range(max_sess_len):
            raw = mask[:, i, :]
            add = mask.masked_fill(~(same[:, i, :].unsqueeze(-1)), 0).sum(1).bool()#bigbatch,
            mask[:, i, :] = raw | add
        mask = mask.masked_fill(repeat.unsqueeze(-1), False)

        atten = mask.long() * torch.softmax(unmaksed_atten.masked_fill(~mask, -1e10),
                                                  2)  #n_head* batch, max_sess_len, 2*max_sess_len*3

        # attn: n_head* batch, max_sess_len, 2*max_sess_len*3;   passage: n_head* batch, 2*max_sess_len*3, dk
        neigh_feats = torch.bmm(atten, pass_message).view(self.n_heads, batch, max_sess_len, self.d_k
                                                          ).transpose(0, 1).transpose(2, 1).contiguous().view(batch * max_sess_len,
                                                                                              -1)  # batch * max_sess_len, n_head * dk
        neigh_feats = F.gelu(neigh_feats)
        retype = self.drop(self.get_type_emb(neigh_feats, [type] * big_batch, qkv='a'))  # batch*session_len, hidden
        output_embed = self.update(retype, centers, [type] * big_batch).view(batch, max_sess_len, self.input_dim)#
        
        return output_embed, repeat_

    def session_graph(self, query_emb, session_qids, session_dids, session_len):
        # session_qids: batch, max_session_len
        # seseion_len: batch
        win_len = 1
        max_sess_len = session_qids.size(1)
        batch = session_qids.size(0)

        device = session_qids.device

        # batch, session_len
        mask = torch.arange(max_sess_len).to(device).unsqueeze(0).repeat([batch, 1])
        mask_l = mask + win_len
        mask_r = mask - win_len
        session_mask = torch.where(mask < session_len.unsqueeze(-1), 1, 0)  # batch, session_len
        pre_index = torch.arange(max_sess_len).unsqueeze(0).repeat([batch, 1]).to(device)  # batch, session_len
        # We use <= and >= rather than > and < because we need to let the node itself as its neighbor of each edge
        blank = torch.zeros([batch, max_sess_len, max_sess_len]).to(device)
        pre_mask = torch.where(mask.unsqueeze(1).repeat([1, max_sess_len, 1]) >= pre_index.unsqueeze(-1), 0, 1) \
                   * session_mask.unsqueeze(1) * session_mask.unsqueeze(-1)  # batch, session_len, session_len
        #pre_mask_l = torch.where(mask_l.unsqueeze(1).repeat([1, max_sess_len, 1]) >= pre_index.unsqueeze(-1), 1, 0)
        #pre_mask = pre_mask * pre_mask_l * session_mask.unsqueeze(1) * session_mask.unsqueeze(-1)
        
        pre_edge_mask = pre_mask.sum(2) == 0
        back_mask = torch.where(mask.unsqueeze(1).repeat([1, max_sess_len, 1]) > pre_index.unsqueeze(-1), 1, 0) \
                    * session_mask.unsqueeze(1) * session_mask.unsqueeze(-1)  # batch, session_len, session_len
        #back_mask_r = torch.where(mask_r.unsqueeze(1).repeat([1, max_sess_len, 1]) > pre_index.unsqueeze(-1), 0, 1)
        #back_mask = back_mask * back_mask_r * session_mask.unsqueeze(1) * session_mask.unsqueeze(-1)

        back_edge_mask = back_mask.sum(2) == 0
        click_mask = torch.eye(max_sess_len).unsqueeze(0).repeat([batch, 1, 1]).to(device)# batch, max_sess_len, max_sess_len
        
        input = torch.cat([session_qids.view(batch * max_sess_len), session_dids.view(batch * max_sess_len)], dim=0)
        indices = input != 0
        all_embs = torch.zeros([input.size(0), 768]).to(device)
        all_embs[indices] = self.features(input[indices])  # We only use the encoder1 to produce the search intent, whose feature is bert

        querys_embs = all_embs[:batch * max_sess_len]#.view(batch, max_sess_len, -1)
        docs_embs = all_embs[batch * max_sess_len:]#.view(batch, max_sess_len, -1)  # batch * session_len, hidden_size
        
        output_embed_q, _ = self.session_graph_message(session_qids, querys_embs, all_embs, pre_mask, back_mask, click_mask, blank, batch, max_sess_len, type=0)
        #all_embs[:batch * max_sess_len] = output_embed_q.view(batch * max_sess_len, -1)
        all_embs_new = torch.cat([output_embed_q.view(batch * max_sess_len, -1), docs_embs], dim=0)
        output_embed_d, repeat = self.session_graph_message(session_dids, docs_embs, all_embs_new, pre_mask, back_mask, click_mask, blank, batch, max_sess_len, type=1)
        #repeat: batch, session_len(bool). True means this doc node is repeated
        
        pos = torch.arange(max_sess_len).unsqueeze(0).repeat([batch, 1]).long().to(device)
        pos_embs = self.pos_embedding(pos)#batch, session_len, hidden
        output_embed_d_pos = self.sess_mlp(torch.cat([output_embed_d, pos_embs], dim=-1))#batch, sesslen, hidden

        sess_atten = self.sess_a(F.gelu(self.sess_w(torch.cat([query_emb.unsqueeze(1).repeat([1, max_sess_len, 1]),
                               output_embed_d_pos], dim=-1))))#batch, sesslen, 1

        session_mask = session_mask * (~repeat).long()
        sess_atten = torch.softmax(sess_atten.masked_fill(~(session_mask.unsqueeze(-1).bool()), -1e10), dim=1)
        session_intent = (sess_atten * output_embed_d).sum(1)

        return session_intent

    def forward(self, nodes, step, loc=None):
        device = nodes.device
        nodes_numpy = nodes.detach().cpu().numpy()
        samp_neighs_lists = self.sample_neighs_all(nodes, nodes_numpy, self.sample_size).detach().cpu().numpy().tolist()

        flat_neigh, node2ind = self.exclusive_combine3(samp_neighs_lists)
        embed_matrix = self.features(torch.LongTensor(flat_neigh).to(device), step+1)  # all_cnt, hidden

        return self.message_process(nodes, samp_neighs_lists, flat_neigh, node2ind, embed_matrix, step)

    def message_process(self, nodes, samp_neighs_lists, flat_neigh, node2ind, embed_matrix, step):
        # the simplest aggregate, for each edge type passage message like GCN and atten add all type embedding
        # self.neighbor_dict is a list, each type edge contains a neighbor dict
        #nodes: batch,
        device = nodes.device
        batch = nodes.size(0)
        #print('batch = ', batch)
        all_cnt = embed_matrix.size(0)

        '''
        deal end
        '''
        neigh_types = [self.nodetype_dict[nei] for nei in flat_neigh]  # get the node type of all related neighbors
        center_types = [self.nodetype_dict[int(nei)] for nei in nodes]  # get the node type of center nodes
        node_index = [node2ind[int(node)] for node in nodes]  # node2ind[nodes]

        # The mask for aggregation for each edge
        '''
        deal with the linear neighbor attention
        '''

        # neigh_feats = torch.zeros(nodes.size(0), self.edge_num + 1, self.input_dim).cuda()
        #n_head, batch, d_k
        q = self.get_type_emb(embed_matrix[node_index], center_types, qkv='q').view(batch, self.n_heads, self.d_k).transpose(0,1).contiguous()
        # n_head, all_cnt, d_k
        k = self.get_type_emb(embed_matrix, neigh_types, qkv='k').view(all_cnt, self.n_heads, self.d_k).transpose(0,1).contiguous()
        v = self.get_type_emb(embed_matrix, neigh_types, qkv='v').view(all_cnt, self.n_heads, self.d_k).transpose(0,1).contiguous()

        '''
        k_proj = torch.bmm(k.unsqueeze(1).repeat([1, self.edge_num, 1, 1]).view(self.n_heads * self.edge_num, all_cnt, self.d_k),
                  self.relation_att)#head * edge_num, all_cnt, dk
        atten = torch.bmm(q.unsqueeze(1).repeat([1, self.edge_num, 1, 1]).view(self.n_heads*self.edge_num, batch, self.d_k),
                  k_proj.transpose(1, 2).contiguous()).view(self.n_heads, self.edge_num, batch, all_cnt)
        '''

        mask = Tensor2Varible(torch.zeros(nodes.size(0), len(flat_neigh), self.edge_num, requires_grad=False))
        pass_message = []
        unmaksed_atten = []
        for i in range(self.edge_num):
            samp_neighs = samp_neighs_lists[i]
            # mask = Tensor2Varible(torch.zeros(nodes.size(0), len(flat_neigh), requires_grad=False))
            # The connections
            column_indices = [node2ind[int(n)] for samp_neigh in samp_neighs for n in samp_neigh]
            row_indices = [i for i in range(len(samp_neighs)) for j in range(len(samp_neighs[i]))]
            mask[row_indices, column_indices, i] = 1
            self_column_indices = [node2ind[int(n)] for n in nodes]
            self_row_indices = [i for i in range(len(samp_neighs))]
            mask[self_row_indices, self_column_indices, i] = 0  # cut self-loop

            k_proj = torch.bmm(k, self.relation_att[i])  # head, all_cnt, dk

            #self.n_heads, batch, all_cnt
            atten_i = torch.bmm(q, k_proj.transpose(1, 2).contiguous()) * self.relation_pri[i].unsqueeze(-1).unsqueeze(-1) / self.sqrt_dk
            unmaksed_atten.append(atten_i.unsqueeze(-1))#n_head, batch, all_cnt, 1

            pass_message.append(torch.bmm(v, self.relation_msg[i]).unsqueeze(2))#n_head, all_cnt, 1,  d_k

        pass_message = torch.cat(pass_message, dim=2).view(self.n_heads, all_cnt*self.edge_num, self.d_k)  # n_head, all_cnt*8, dk
        unmaksed_atten = torch.cat(unmaksed_atten, dim=3).view(self.n_heads, batch, all_cnt*self.edge_num) #n_head, batch, all_cnt*8
        mask = mask.view(batch, all_cnt * self.edge_num)
        atten = mask.unsqueeze(0) * torch.softmax(unmaksed_atten.masked_fill((1 - mask.unsqueeze(0)).bool(), -1e10), 2)  # n_head, batch, all_cnt*8

        # attn: n_head, batch, all_cnt*8;   passage: n_head, all_cnt*8, dk
        neigh_feats = torch.bmm(atten, pass_message).view(self.n_heads, batch, self.d_k
                                       ).transpose(0, 1).contiguous().view(batch, -1)  # batch, d_k * head(hidden)

        neigh_feats = F.gelu(neigh_feats)
        retype = self.drop(self.get_type_emb(neigh_feats, center_types, qkv='a'))#batch, hidden
        output_embed = self.update(retype, embed_matrix[node_index], center_types)
        if step == 2: output_embed = self.drop(output_embed)

        return output_embed  # node_num * text_dim

    def update(self, aggr_out, node_inp, node_type):
        '''
            Step 3: Target-specific Aggregation
            x = W[node_type] * gelu(Agg(x)) + x
        '''
        alpha = torch.sigmoid(self.skip[node_type]).unsqueeze(-1)#batch
        node_type = torch.LongTensor(node_type).to(aggr_out.device)
        res = (aggr_out * alpha + node_inp * (1-alpha))#batch, hidden

        skip_idx = (aggr_out == 0).long().sum(-1) == aggr_out.size(1)#this node have no neighbors
        if self.use_norm:
            res[node_type.bool()] = self.norms[1](res[node_type.bool()])
            res[(1-node_type).bool()] = self.norms[0](res[(1-node_type).bool()])
        res[skip_idx] = node_inp[skip_idx]
        
        res = self.drop(self.out_linear(F.gelu(self.mid_linear(res)))) + res
        res = self.out_norm(res)
        
        return res

class BERT_encoder(nn.Module):
    def __init__(self, config):
        super(BERT_encoder, self).__init__()
        self.id2emb = None
        self.load_embedding(config['bert_id2emb_dict'])

        self.dropout = nn.Dropout(config['bert_drt'])
        print('id2emb size = ', self.id2emb.shape)

    def load_embedding(self, path):
        embedding_list = []
        for i in range(6):
            embedding_list.append(pickle.load(open(path.format(i), 'rb')))
        self.id2emb = numpy.vstack(embedding_list)

    def forward(self, nodes_list, step=None):
        device = nodes_list.device
        embeddings = torch.Tensor(self.id2emb[nodes_list.detach().cpu().numpy()]).to(device)
        return self.dropout(embeddings)
