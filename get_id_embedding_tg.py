import pickle
import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer
from tqdm import tqdm
import numpy as np
from RankModel_pretower_contra import Ranker_nobert
from config import basic_config_tg

device = torch.device("cuda:0")
path = r'./process_data_tg/id2textid.base.pkl'
save_path = r'./process_data_tg/id2emb_contra{}_base_bt100.pkl'
id2type = pickle.load(open('./process_data_tg/id_type.base.pkl', 'rb'))
query2id = pickle.load(open('./process_data_tg/query2id.base.pkl', 'rb'))
doc2id = pickle.load(open('./process_data_tg/doc2id.base.pkl', 'rb'))
config_state = basic_config_tg()
#config_state['hidden_dim'] = 64

id2input = np.zeros((len(id2type)+1, 128))
id2atten = np.zeros((len(id2type)+1, 128))
id2token = np.zeros((len(id2type)+1, 128))

tokenizer = BertTokenizer.from_pretrained('./bert-base-chinese')
additional_tokens = 4
tokenizer.add_tokens("[eos]")
tokenizer.add_tokens("[empty_d]")
tokenizer.add_tokens("[term_del]")
tokenizer.add_tokens("[sent_del]")


for query in tqdm(query2id):
    inputs = tokenizer.encode_plus(query, max_length=128, truncation=True, padding='max_length')
    id2input[query2id[query]] = inputs['input_ids']
    id2atten[query2id[query]] = inputs['attention_mask']
    id2token[query2id[query]] = inputs['token_type_ids']
    
for doc in tqdm(doc2id):
    inputs = tokenizer.encode_plus(doc, max_length=128, truncation=True, padding='max_length')
    id2input[doc2id[doc]] = inputs['input_ids']
    id2atten[doc2id[doc]] = inputs['attention_mask']
    id2token[doc2id[doc]] = inputs['token_type_ids']

pickle.dump(id2input, open('id2input.pkl', 'wb'))
pickle.dump(id2atten, open('id2atten.pkl', 'wb'))
pickle.dump(id2token, open('id2token.pkl', 'wb'))

model = Ranker_nobert(additional_tokens, config_state)
model.load_state_dict(torch.load(r'./model/nobert.tiangong.4e-5.100.nodeemb.contra'))
model.to(device)

'''
bert_q = model.bert_model_base_q.to(device)
bert_d = model.bert_model_base_d.to(device)

pooler = model.pooler#rnn.Linear(768, self.hidden_dim)
activation = nn.Tanh()
bert_q.eval()
bert_d.eval()
'''

model.eval()

MAX_LEN = 128
batch_size = 100

def get_inputs(ids):
    return torch.LongTensor(id2input[ids]).to(device), torch.LongTensor(id2atten[ids]).to(device), torch.LongTensor(id2token[ids]).to(device)

def get_emb(input_ids, token_type_ids, attention_mask, type=0):
    inputs = {'input_ids': input_ids, 'attention_mask': attention_mask, 'token_type_ids': token_type_ids}
    if type == 0:
        emb = model.bert_model_base_q(**inputs)[1]
    else:
        emb = model.bert_model_base_d(**inputs)[1]
    return emb.detach().cpu().numpy()
    
        
def get_embeddings():

    id2embed = np.zeros([len(id2type)+1, 768])
    
    cnt_q = 0
    cnt_d = 0
    ind_q = []
    ind_d = []
    
    bid = 1
    eid = batch_size + 1
    

    for ids in tqdm(id2type):
        if id2type[ids] == 0: #query
            cnt_q += 1
            ind_q.append(ids)
        else:
            cnt_d += 1
            ind_d.append(ids)
        
        '''
        cls = cls[0,:].detach().cpu().numpy()
        assert cls.shape[0] == 768
        id2embed[ids,:] = cls
        '''
        
        if cnt_q == batch_size:
            '''
            out_layers, cls = bert_q(input_ids=torch.LongTensor(input_ids_q).to(device),
                                   token_type_ids=torch.LongTensor(token_type_ids_q).to(device),
                                   attention_mask=torch.LongTensor(attention_mask_q).to(device)
                                   )[:]
            #print('out_layers size = ', out_layers.size())
            #print('cls size = ', cls.size())
            cls = cls.detach().cpu().numpy()
            last_emb = out_layers.detach().cpu().numpy()[:,0,:]
            '''
            input_ids, attention_mask, token_type = get_inputs(ind_q)
            emb = get_emb(input_ids, token_type, attention_mask, type=0)
            id2embed[np.array(ind_q, dtype=np.int32), :] = emb
            cnt_q = 0
            ind_q = []
        if cnt_d == batch_size:
            '''
            out_layers, cls = bert_d(input_ids=torch.LongTensor(input_ids_d).to(device),
                                   token_type_ids=torch.LongTensor(token_type_ids_d).to(device),
                                   attention_mask=torch.LongTensor(attention_mask_d).to(device)
                                   )[:]
            #print('out_layers size = ', out_layers.size())
            #print('cls size = ', cls.size())
            cls = cls.detach().cpu().numpy()
            last_emb = out_layers.detach().cpu().numpy()[:,0,:]
            '''
            input_ids, attention_mask, token_type = get_inputs(ind_d)
            emb = get_emb(input_ids, token_type, attention_mask, type=1)
            id2embed[np.array(ind_d, dtype=np.int32), :] = emb
            cnt_d = 0
            ind_d = []
            '''
            bid += batch_size
            eid += batch_size
            '''
        '''
        text = id2text[ids]
        #row text ids contain [cls] [sep], thus [1:-1] to cut the twice adding
        inputs = tokenizer.encode_plus(text)
        input_ids[cnt, :len(inputs['input_ids'])-2] = inputs['input_ids'][1:-1]
        token_type_ids[cnt, :len(inputs['input_ids'])-2] = inputs['token_type_ids'][1:-1]
        attention_mask[cnt, :len(inputs['input_ids'])-2] = inputs['attention_mask'][1:-1]

        cnt += 1
        '''
    
    if cnt_q > 0:
        '''
        out_layers, cls = bert_q(input_ids=torch.LongTensor(input_ids_q[:cnt_q]).to(device),
                           token_type_ids=torch.LongTensor(token_type_ids_q[:cnt_q]).to(device),
                           attention_mask=torch.LongTensor(attention_mask_q[:cnt_q]).to(device)
                           )[:]
        cls = cls.detach().cpu().numpy()
        last_emb = out_layers.detach().cpu().numpy()[:,0,:]
        '''
        input_ids, attention_mask, token_type = get_inputs(ind_q)
        emb = get_emb(input_ids, token_type, attention_mask, type=0)
        id2embed[np.array(ind_q, dtype=np.int32), :] = emb
        cnt_q = 0
        ind_q = []
    if cnt_d > 0:
        '''
        out_layers, cls = bert_d(input_ids=torch.LongTensor(input_ids_d[:cnt_d]).to(device),
                                   token_type_ids=torch.LongTensor(token_type_ids_d[:cnt_d]).to(device),
                                   attention_mask=torch.LongTensor(attention_mask_d[:cnt_d]).to(device)
                                   )[:]
        cls = cls.detach().cpu().numpy()
        last_emb = out_layers.detach().cpu().numpy()[:,0,:]
        '''
        input_ids, attention_mask, token_type = get_inputs(ind_d)
        emb = get_emb(input_ids, token_type, attention_mask, type=1)
        id2embed[np.array(ind_d, dtype=np.int32), :] = emb
        cnt_d = 0
        ind_d = []
    #cls = cls.detach().cpu().numpy()
    #id2embed[bid:eid, :] = cls
    '''
    print('out_layers size = ', out_layers.size())
    print('bid = {} eid = {}'.format(bid, eid))
    last_emb = out_layers.detach().cpu().numpy()[:,0,:]
    id2embed[bid:eid, :] = last_emb
    '''


    length = id2embed.shape[0]

    cut = 5
    percnt = int(length / cut)
    print('id2emb[0] = ', id2embed[0])
    print('id2emb[-1] = ', id2embed[-1])

    for i in range(cut+1):
        if i < cut:
            pickle.dump(id2embed[i*percnt: (i+1)*percnt], open(save_path.format(i), 'wb'))
        else:
            pickle.dump(id2embed[i * percnt: ], open(save_path.format(i), 'wb'))


    # with open(save_path, 'w') as f:
    #     for i in range(id2embed.shape[0]):
    #         text_list = '\t'.join([str(item) for item in id2embed[i].tolist()]) + '\n'
    #
    #         f.write(text_list)

    # pickle.dump(id2embed, open(save_path, 'wb'))

get_embeddings()


