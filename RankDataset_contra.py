import torch
import linecache
from torch.utils.data import Dataset
import numpy as np
import random
import re

class RankDataset_contra(Dataset):
    def __init__(self, filename, max_q_length, max_d_length, tokenizer):
        super(RankDataset_contra, self).__init__()
        self._filename = filename
        self._max_q_length = max_q_length
        self._max_d_length = max_d_length
        self._max_length = max(max_q_length, max_d_length)
        self._tokenizer = tokenizer
        with open(filename, "r") as f:
            self._total_data = len(f.readlines())
            
    def get_inputs(self, strs, max_len):
        inputs = self._tokenizer.encode_plus(strs, max_length=max_len, truncation=True, padding='max_length', return_tensors='pt')
        return inputs['input_ids'][0], inputs['attention_mask'][0], inputs['token_type_ids'][0]
        
    def anno_main(self, qd_pairs):
        #['qd', 'qd',...]
        all_qd = []
        for qd in qd_pairs:
            qd = self._tokenizer.tokenize(qd)# get the word list(haven't encoded as the token ids)
            all_qd.append(qd)
        all_qd = self.check_length(all_qd)
        history = all_qd[:-2]
        query_tok = all_qd[-2]
        doc_tok = all_qd[-1]
        history_toks = ["[CLS]"]
        segment_ids = [0]
        for iidx, sent in enumerate(history):
            history_toks.extend(sent + ["[eos]"])#[[q,d],[q,d],...]-->[q,d,q,d,...]
            segment_ids.extend([0] * (len(sent) + 1))
        query_tok += ["[eos]"]
        query_tok += ["[SEP]"]
        doc_tok += ["[eos]"]
        doc_tok += ["[SEP]"]
        all_qd_toks = history_toks + query_tok + doc_tok
        segment_ids.extend([0] * len(query_tok))
        segment_ids.extend([0] * len(doc_tok))
        all_attention_mask = [1] * len(all_qd_toks)
        assert len(all_qd_toks) <= self._max_seq_length
        while len(all_qd_toks) < self._max_seq_length:
            all_qd_toks.append("[PAD]")
            segment_ids.append(0)
            all_attention_mask.append(0)
        assert len(all_qd_toks) == len(segment_ids) == len(all_attention_mask) == self._max_seq_length
        anno_seq = self._tokenizer.convert_tokens_to_ids(all_qd_toks)
        input_ids = np.asarray(anno_seq)
        all_attention_mask = np.asarray(all_attention_mask)
        segment_ids = np.asarray(segment_ids)
        return input_ids, all_attention_mask, segment_ids

    def __getitem__(self, idx):
        line = linecache.getline(self._filename, idx + 1)
        line = line.strip().split("\t")
        
        label = int(line[0])
        length = len(line)-1
        assert length & 1 == 0
        length = int(length / 2)
        # print('length = ', length)
        qd_pairs = line[1:1+length]  # ['qd', 'qd', ...] the sequence information including the current query and document
        
        query = qd_pairs[-2]
        doc = qd_pairs[-1]
        input_ids_q, attention_mask_q, token_type_ids_q = self.get_inputs(query, self._max_q_length)
        input_ids_d, attention_mask_d, token_type_ids_d = self.get_inputs(doc, self._max_d_length)

        batch = {
            'input_ids_q': input_ids_q,
            'token_type_ids_q': token_type_ids_q,
            'attention_mask_q': attention_mask_q,
            'input_ids_d': input_ids_d,
            'token_type_ids_d': token_type_ids_d,
            'attention_mask_d': attention_mask_d,
        }
        return batch


    def __len__(self):
        return self._total_data
        
class RankDataset_contra_test(Dataset):
    def __init__(self, filename, max_q_length, max_d_length, tokenizer):
        super(RankDataset_contra_test, self).__init__()
        self._filename = filename
        self._max_q_length = max_q_length
        self._max_d_length = max_d_length
        self._max_length = max(max_q_length, max_d_length)
        self._tokenizer = tokenizer
        with open(filename, "r") as f:
            self._total_data = len(f.readlines())
            
    def get_inputs(self, strs, max_len):
        inputs = self._tokenizer.encode_plus(strs, max_length=max_len, truncation=True, padding='max_length', return_tensors='pt')
        return inputs['input_ids'][0], inputs['attention_mask'][0], inputs['token_type_ids'][0]
        
    def anno_main(self, qd_pairs):
        #['qd', 'qd',...]
        all_qd = []
        for qd in qd_pairs:
            qd = self._tokenizer.tokenize(qd)# get the word list(haven't encoded as the token ids)
            all_qd.append(qd)
        all_qd = self.check_length(all_qd)
        history = all_qd[:-2]
        query_tok = all_qd[-2]
        doc_tok = all_qd[-1]
        history_toks = ["[CLS]"]
        segment_ids = [0]
        for iidx, sent in enumerate(history):
            history_toks.extend(sent + ["[eos]"])#[[q,d],[q,d],...]-->[q,d,q,d,...]
            segment_ids.extend([0] * (len(sent) + 1))
        query_tok += ["[eos]"]
        query_tok += ["[SEP]"]
        doc_tok += ["[eos]"]
        doc_tok += ["[SEP]"]
        all_qd_toks = history_toks + query_tok + doc_tok
        segment_ids.extend([0] * len(query_tok))
        segment_ids.extend([0] * len(doc_tok))
        all_attention_mask = [1] * len(all_qd_toks)
        assert len(all_qd_toks) <= self._max_seq_length
        while len(all_qd_toks) < self._max_seq_length:
            all_qd_toks.append("[PAD]")
            segment_ids.append(0)
            all_attention_mask.append(0)
        assert len(all_qd_toks) == len(segment_ids) == len(all_attention_mask) == self._max_seq_length
        anno_seq = self._tokenizer.convert_tokens_to_ids(all_qd_toks)
        input_ids = np.asarray(anno_seq)
        all_attention_mask = np.asarray(all_attention_mask)
        segment_ids = np.asarray(segment_ids)
        return input_ids, all_attention_mask, segment_ids

    def __getitem__(self, idx):
        line = linecache.getline(self._filename, idx + 1)
        line = line.strip().split("\t")
        
        label = int(line[0])
        length = len(line)-1
        assert length & 1 == 0
        length = int(length / 2)
        # print('length = ', length)
        qd_pairs = line[1:1+length]  # ['qd', 'qd', ...] the sequence information including the current query and document
        
        query = qd_pairs[-2]
        doc = qd_pairs[-1]
        input_ids_q, attention_mask_q, token_type_ids_q = self.get_inputs(query, self._max_q_length)
        input_ids_d, attention_mask_d, token_type_ids_d = self.get_inputs(doc, self._max_d_length)

        batch = {
            'input_ids_q': input_ids_q,
            'token_type_ids_q': token_type_ids_q,
            'attention_mask_q': attention_mask_q,
            'input_ids_d': input_ids_d,
            'token_type_ids_d': token_type_ids_d,
            'attention_mask_d': attention_mask_d,
            'label': label
        }
        return batch


    def __len__(self):
        return self._total_data