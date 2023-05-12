import torch
import linecache
from torch.utils.data import Dataset
import numpy as np
import random
import re

class RankDataset_point(Dataset):
    def __init__(self, filename, max_seq_length, max_sess_length, tokenizer):
        super(RankDataset_point, self).__init__()
        self._filename = filename
        self._max_seq_length = max_seq_length
        self._max_sess_length = max_sess_length
        self._tokenizer = tokenizer
        with open(filename, "r") as f:
            self._total_data = len(f.readlines())

    def check_length(self, pairlist):
        assert len(pairlist) % 2 == 0
        max_seq_length = self._max_seq_length - 3
        if len(pairlist) == 2:
            while len(pairlist[0]) + len(pairlist[1]) + 2 > max_seq_length:
                if len(pairlist[0]) > len(pairlist[1]):
                    pairlist[0].pop(0)
                else:
                    pairlist[1].pop(-1)
        else:
            q_d_minimum_length = 0
            for i in range(len(pairlist)):
                q_d_minimum_length += len(pairlist[i]) + 1
            if q_d_minimum_length > max_seq_length:
                pairlist.pop(0)
                pairlist.pop(0)
                pairlist = self.check_length(pairlist)
        return pairlist

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
        # line = line.stripe().split("\t")
        label = int(line[0])
        length = len(line)-1
        assert length & 1 == 0
        length = int(length / 2)
        # print('length = ', length)
        qd_pairs = line[1:1+length]  # ['qd', 'qd', ...] the sequence information including the current query and document
        qd_id_pairs = [int(item) for item in line[1+length:-2]]

        # print('qd paris = {}, qd_id_pairs = {}'.format(qd_pairs, qd_id_pairs))

        assert len(qd_pairs) & 1 == 0
        assert len(qd_id_pairs) & 1 == 0

        input_ids, attention_mask, segment_ids = self.anno_main(qd_pairs)

        qd_id_pairs = qd_id_pairs[-(self._max_sess_length * 2):]
        qid = int(line[-2])
        did = int(line[-1])

        # qd_id_pairs = qd_id_pairs[:-2]

        len_sess = len(qd_id_pairs) // 2
        if len_sess < self._max_sess_length:
            qd_id_pairs += [0, 0] * int(self._max_sess_length - len_sess)

        array_ = np.array(qd_id_pairs).reshape(self._max_sess_length, 2)
        session_qids = array_[:, 0]
        session_qids[len_sess] = qid#add qid
        session_dids = array_[:, 1]
        # print('session qids = {} session dids = {}'.format(session_qids, session_dids))


        batch = {
            'input_ids': input_ids,
            'token_type_ids': segment_ids,
            'attention_mask': attention_mask,
            'session_qid': session_qids,
            'session_did': session_dids,
            'session_len': len_sess + 1,
            'qid': qid,
            'did': did,
            'label': label
        }
        return batch


    def __len__(self):
        return self._total_data


class RankDataset_test(Dataset):
    def __init__(self, candi_cnt,  filename, max_seq_length, max_sess_length, tokenizer):
        super(RankDataset_test, self).__init__()
        self._candi_cnt = candi_cnt
        self._filename = filename
        self._max_seq_length = max_seq_length
        self._max_sess_length = max_sess_length
        self._tokenizer = tokenizer
        with open(filename, "r") as f:
            linecnt = len(f.readlines())
            assert linecnt % self._candi_cnt == 0
            self._total_data = linecnt // self._candi_cnt

    def check_length(self, pairlist):
        assert len(pairlist) % 2 == 0
        max_seq_length = self._max_seq_length - 3
        if len(pairlist) == 2:
            while len(pairlist[0]) + len(pairlist[1]) + 2 > max_seq_length:
                if len(pairlist[0]) > len(pairlist[1]):
                    pairlist[0].pop(0)
                else:
                    pairlist[1].pop(-1)
        else:
            q_d_minimum_length = 0
            for i in range(len(pairlist)):
                q_d_minimum_length += len(pairlist[i]) + 1
            if q_d_minimum_length > max_seq_length:
                pairlist.pop(0)
                pairlist.pop(0)
                pairlist = self.check_length(pairlist)
        return pairlist

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
        # lines = []
        input_ids_all = []
        segment_ids_all = []
        attention_mask_all = []
        did_all = []
        label_all = []
        pre_qid = -1
        pre_sess_qid = -1
        pre_sess_did = -1
        for i in range(self._candi_cnt):
            line = linecache.getline(self._filename, idx * self._candi_cnt + 1 + i)
            line = line.strip().split("\t")
            # line = line.stripe().split("\t")
            label = int(line[0])
            length = len(line)-1
            assert length & 1 == 0
            length = int(length / 2)
            # print('length = ', length)
            qd_pairs = line[1:1+length]  # ['qd', 'qd', ...] the sequence information including the current query and document
            qd_id_pairs = [int(item) for item in line[1+length:-2]]

            # print('qd paris = {}, qd_id_pairs = {}'.format(qd_pairs, qd_id_pairs))

            assert len(qd_pairs) & 1 == 0
            assert len(qd_id_pairs) & 1 == 0

            input_ids, attention_mask, segment_ids = self.anno_main(qd_pairs)

            qd_id_pairs = qd_id_pairs[-(self._max_sess_length * 2):]
            qid = int(line[-2])
            did = int(line[-1])
            # qd_id_pairs = qd_id_pairs[:-2]

            len_sess = len(qd_id_pairs) // 2
            if len_sess < self._max_sess_length:
                qd_id_pairs += [0, 0] * int(self._max_sess_length - len_sess)

            array_ = np.array(qd_id_pairs).reshape(self._max_sess_length, 2)
            session_qids = array_[:, 0]
            session_qids[len_sess] = qid#add qid
            session_dids = array_[:, 1]
            
            if pre_qid == -1: 
                pre_qid = qid
                pre_sess_qid = session_qids
                pre_sess_did = session_dids
                
            else: 
                assert pre_qid == qid
                for i in range(len(pre_sess_qid)):
                    assert pre_sess_qid[i] == session_qids[i]
                    assert pre_sess_did[i] == session_dids[i]
            
            input_ids_all.append(input_ids)
            segment_ids_all.append(segment_ids)
            attention_mask_all.append(attention_mask)
            did_all.append(did)
            label_all.append(label)
        # print('session qids = {} session dids = {}'.format(session_qids, session_dids))


        batch = {
            'input_ids': np.array(input_ids_all, dtype=np.int),
            'token_type_ids': np.array(segment_ids_all, dtype=np.int),
            'attention_mask': np.array(attention_mask_all, dtype=np.int),
            'session_qid': session_qids,
            'session_did': session_dids,
            'session_len': len_sess + 1,
            'qid': qid,
            'did': np.array(did_all, dtype=np.int),
            'label': np.array(label_all, dtype=np.int)
        }
        return batch


    def __len__(self):
        return self._total_data
