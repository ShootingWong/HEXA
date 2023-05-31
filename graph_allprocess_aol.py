import numpy as np
import os
import os.path as osp
import pickle
import json
from tqdm import tqdm
from transformers import BertTokenizer, BertModel
import copy

max_freq = 0
bert_path = r'../../SubPer/bert'

query2id = dict()
id2query = dict()
doc2id = dict()
id2doc = dict()
id2type = dict()

query_uncase = set()
doc_uncase = set()

session_q_graph = dict()
session_d_graph = dict()
click_graph = dict()
top_graph = dict()

click_neb_q = dict()#0
click_neb_d = dict()#1
top_neb_q = dict()#2
top_neb_d = dict()#3
back_neb_q = dict()#4
pre_neb_q = dict()#5
back_neb_d = dict()#6
pre_neb_d = dict()#7

click_neb_q_freq = dict()#0
click_neb_d_freq = dict()#1
top_neb_q_freq = dict()#2
top_neb_d_freq = dict()#3
back_neb_q_freq = dict()#4
pre_neb_q_freq = dict()#5
back_neb_d_freq = dict()#6
pre_neb_d_freq = dict()#7

n_click_neb_q_freq = dict()  # 0
n_click_neb_d_freq = dict()  # 1
n_top_neb_q_freq = dict()  # 2
n_top_neb_d_freq = dict()  # 3
n_back_neb_q_freq = dict()  # 4
n_pre_neb_q_freq = dict()  # 5
n_back_neb_d_freq = dict()  # 6
n_pre_neb_d_freq = dict()  # 7

n_click_neb_q = None
n_top_neb_q = None
n_back_neb_q = None
n_pre_neb_q = None
n_click_neb_d = None
n_top_neb_d = None
n_back_neb_d = None
n_pre_neb_d = None

tokenizer = BertTokenizer.from_pretrained(bert_path)

id2textid = dict()
ids_train = set()

q_none_id = -1 # id of ''
d_none_id = -1 # id of ''(not [empty]!!)
ind = 1

def get_neighbor(node, edgetype, neigh_freq_list, father=None):
    if father is None:
        return copy.deepcopy(neigh_freq_list[edgetype][node])
    else:
        dicts = copy.deepcopy(neigh_freq_list[edgetype][node])
        if father in dicts:
            del dicts[father]
        return dicts

def add_neighbor(v1, v2, type=0):
    if type == 0:
        #q d click
        # if v1 not in click_neb_q:
        #     click_neb_q[v1] = set()
        if v2 not in click_neb_q[v1]:
            click_neb_q[v1].add(v2)
            click_neb_q_freq[v1][v2] = 1
        else:
            click_neb_q_freq[v1][v2] += 1

    elif type == 1:
        # if v1 not in click_neb_d:
        #     click_neb_d[v1] = set()
        if v2 not in click_neb_d[v1]:
            click_neb_d[v1].add(v2)
            click_neb_d_freq[v1][v2] = 1
        else:
            click_neb_d_freq[v1][v2] += 1
    elif type == 2:
        # if v1 not in top_neb_q:
        #     top_neb_q[v1] = set()
        if v2 not in top_neb_q[v1]:
            top_neb_q[v1].add(v2)
            top_neb_q_freq[v1][v2] = 1
        else:
            top_neb_q_freq[v1][v2] += 1
    elif type == 3:
        # if v1 not in top_neb_d:
        #     top_neb_d[v1] = set()
        if v2 not in top_neb_d[v1]:
            top_neb_d[v1].add(v2)
            top_neb_d_freq[v1][v2] = 1
        else:
            top_neb_d_freq[v1][v2] += 1
    elif type == 4:
        # if v1 not in back_neb_q:
        #     back_neb_q[v1] = set()
        if v2 not in back_neb_q[v1]:
            back_neb_q[v1].add(v2)
            back_neb_q_freq[v1][v2] = 1
        else:
            back_neb_q_freq[v1][v2] += 1

    elif type == 5:
        # if v1 not in pre_neb_q:
        #     pre_neb_q[v1] = set()
        if v2 not in pre_neb_q[v1]:
            pre_neb_q[v1].add(v2)
            pre_neb_q_freq[v1][v2] = 1
        else:
            pre_neb_q_freq[v1][v2] += 1
    elif type == 6:
        # if v1 not in back_neb_d:
        #     back_neb_d[v1] = set()
        if v2 not in back_neb_d[v1]:
            back_neb_d[v1].add(v2)
            back_neb_d_freq[v1][v2] = 1
        else:
            back_neb_d_freq[v1][v2] += 1
    elif type == 7:
        # if v1 not in pre_neb_d:
        #     pre_neb_d[v1] = set()
        if v2 not in pre_neb_d[v1]:
            pre_neb_d[v1].add(v2)
            pre_neb_d_freq[v1][v2] = 1
        else:
            pre_neb_d_freq[v1][v2] += 1

def print_statistic():
    cnq_list = [len(item) for item in list(click_neb_q.values())]
    cnd_list = [len(item) for item in list(click_neb_d.values())]
    tnq_list = [len(item) for item in list(top_neb_q.values())]
    tnd_list = [len(item) for item in list(top_neb_d.values())]
    pnq_list = [len(item) for item in list(pre_neb_q.values())]
    pnd_list = [len(item) for item in list(pre_neb_d.values())]
    bnq_list = [len(item) for item in list(back_neb_q.values())]
    bnd_list = [len(item) for item in list(back_neb_d.values())]

    print('max click q: ', sorted(cnq_list)[::-1][:50])
    print('max click d: ', sorted(cnd_list)[::-1][:50])
    print('max top q : ', sorted(tnq_list)[::-1][:50])
    print('max top d : ', sorted(tnd_list)[::-1][:50])
    print('max pre q : ', sorted(pnq_list)[::-1][:50])
    print('max pre d : ', sorted(pnd_list)[::-1][:50])
    print('max back q : ', sorted(bnq_list)[::-1][:50])
    print('max back d : ', sorted(bnd_list)[::-1][:50])

    # print('max pre link of query is ', id2query[list(pre_neb_q.keys())[np.argmax(pnq_list)]])
    # print('max pre link of docs is ', id2doc[list(pre_neb_d.keys())[np.argmax(pnd_list)]])
    # print('max back link of query is ', id2query[list(back_neb_q.keys())[np.argmax(bnq_list)]])
    # print('max back link of docs is ', id2doc[list(back_neb_d.keys())[np.argmax(bnd_list)]])
    # print('click q cnt : ', sorted(cnd_list)[::-1])
    print('CLICK Q : MAX:{} | MIN:{} | AVG:{}\nCLICK D : MAX:{} | MIN:{} | AVG:{}\nTOP Q : MAX:{} | MIN:{} | AVG:{}\nTOP D : MAX:{} | MIN:{} | AVG:{}'.format(
        max(cnq_list), min(cnq_list), np.mean(cnq_list),
        max(cnd_list), min(cnd_list), np.mean(cnd_list),
        max(tnq_list), min(tnq_list), np.mean(tnq_list),
        max(tnd_list), min(tnd_list), np.mean(tnd_list)
    ))
    print('\nPRE Q : MAX:{} | MIN:{} | AVG:{}\nPRE D : MAX:{} | MIN:{} | AVG:{}\nBACK Q : MAX:{} | MIN:{} | AVG:{}\nBACK D : MAX:{} | MIN:{} | AVG:{}'.format(
        max(pnq_list), min(pnq_list), np.mean(pnq_list),
        max(pnd_list), min(pnd_list), np.mean(pnd_list),
        max(bnq_list), min(bnq_list), np.mean(bnq_list),
        max(bnd_list), min(bnd_list), np.mean(bnd_list),
    ))

def construct_ind(data_path, train=True):

    global ind, query2id, id2query, id2type, id2texid, click_neb_q, top_neb_q, pre_neb_q, back_neb_q, click_neb_q_freq, \
        top_neb_q_freq, pre_neb_q_freq, back_neb_q_freq, doc2id, id2doc, click_neb_d, top_neb_d, pre_neb_d, back_neb_d, \
        click_neb_d_freq, top_neb_d_freq, pre_neb_d_freq, back_neb_d_freq, click_graph, top_graph, session_q_graph, \
        session_d_graph, ids_train
    DOC_MAX_LEN = 20
    with open(data_path, 'r') as f:
        lines = f.readlines()
        for line in tqdm(lines):
            #each line contains one session information
            data_ = json.loads(line.strip())
            sess_qid = []
            sess_did = []
            for query in data_["query"]:
                q_txt = (" ".join(query["text"].split())).strip()

                q_text_id = tokenizer.encode(q_txt)
                query_uncase.add(q_txt)
                if q_txt not in query2id:
                    query2id[q_txt] = ind
                    id2query[ind] = q_txt
                    # qf.write(q_txt + '\t' + str(ind))
                    id2type[ind] = 0
                    id2textid[ind] = q_text_id
                    ind += 1

                qid = query2id[q_txt]
                
                sess_qid.append(qid)
                if qid not in click_neb_q:
                    click_neb_q[qid] = set()
                    top_neb_q[qid] = set()
                    pre_neb_q[qid] = set()
                    back_neb_q[qid] = set()

                    click_neb_q_freq[qid] = dict()
                    top_neb_q_freq[qid] = dict()
                    pre_neb_q_freq[qid] = dict()
                    back_neb_q_freq[qid] = dict()

                for doc in query["clicks"]:
                    d_txt = (" ".join(doc["title"].split()[:DOC_MAX_LEN])).strip()
                    d_text_id = tokenizer.encode(d_txt)
                    doc_uncase.add(d_txt)
                    if d_txt not in doc2id:
                        doc2id[d_txt] = ind
                        id2doc[ind] = d_txt
                        id2type[ind] = 1
                        id2textid[ind] = d_text_id
                        ind += 1

                    did = doc2id[d_txt]
                    inter_str = str(qid) + ' ' + str(did)
                    if did not in click_neb_d:
                        click_neb_d[did] = set()
                        top_neb_d[did] = set()
                        pre_neb_d[did] = set()
                        back_neb_d[did] = set()

                        click_neb_d_freq[did] = dict()
                        top_neb_d_freq[did] = dict()
                        pre_neb_d_freq[did] = dict()
                        back_neb_d_freq[did] = dict()

                    if doc["label"] == True:
                        sess_did.append(did)
                        if train:
                            if inter_str not in click_graph:
                                click_graph[inter_str] = 1
                            else:
                                click_graph[inter_str] += 1

                            add_neighbor(qid, did, type=0)
                            add_neighbor(did, qid, type=1)
                    else:
                        if train:
                            if inter_str not in top_graph:
                                top_graph[inter_str] = 1
                            else:
                                top_graph[inter_str] += 1
                            add_neighbor(qid, did, type=2)
                            add_neighbor(did, qid, type=3)

            if train:
                for i in range(len(sess_qid)):
                    for j in range(i+1, len(sess_qid)):
                        qinter_str = str(sess_qid[i]) + ' ' + str(sess_qid[j])

                        if qinter_str not in session_q_graph:
                            session_q_graph[qinter_str] = 1
                        else:
                            session_q_graph[qinter_str] += 1
                        add_neighbor(sess_qid[i], sess_qid[j], type=4)#back session q
                        add_neighbor(sess_qid[j], sess_qid[i], type=5)#pre session q

                for i in range(len(sess_did)):
                    for j in range(i+1, len(sess_did)):
                        dinter_str = str(sess_did[i]) + ' ' + str(sess_did[j])
                        if dinter_str not in session_d_graph:
                            session_d_graph[dinter_str] = 1
                        else:
                            session_d_graph[dinter_str] += 1

                        add_neighbor(sess_did[i], sess_did[j], type=6)  # back session d
                        add_neighbor(sess_did[j], sess_did[i], type=7)  # pre session d
    if train:
        ids_train = list(id2type.keys())
    print('Num query:{}(uncase:{})\nNum doc:{}(uncase{})\nNum click: {}\nNum top: {}\nNum q session:{}\nNum d session:{}\n'.\
          format(len(query2id), len(query_uncase), len(doc2id), len(doc_uncase), len(click_graph), len(top_graph), len(session_q_graph), len(session_d_graph)))

    if train:
        print_statistic()

def get_edge_freq():
    global ind, query2id, id2query, id2type, id2texid, click_neb_q, top_neb_q, pre_neb_q, back_neb_q, click_neb_q_freq, \
        top_neb_q_freq, pre_neb_q_freq, back_neb_q_freq, doc2id, id2doc, click_neb_d, top_neb_d, pre_neb_d, back_neb_d, \
        click_neb_d_freq, top_neb_d_freq, pre_neb_d_freq, back_neb_d_freq, click_graph, top_graph, session_q_graph, \
        session_d_graph, ids_train, n_click_neb_q, n_top_neb_q, n_back_neb_q, n_pre_neb_q, n_click_neb_d, n_top_neb_d, \
        n_back_neb_d, n_pre_neb_d, n_click_neb_q_freq, n_click_neb_d_freq, n_top_neb_q_freq, n_top_neb_d_freq, \
        n_back_neb_q_freq, n_pre_neb_q_freq, n_back_neb_d_freq, n_pre_neb_d_freq
    print('Begin to get_edge_freq')
    max_id = max(max(query2id.values()), max(doc2id.values())) + 1

    for node in tqdm(click_neb_q):

        n_click_neb_q_freq[node] = [0]*len(click_neb_q[node])
        n_top_neb_q_freq[node] = [0]*len(top_neb_q[node])
        n_back_neb_q_freq[node] = [0]*len(back_neb_q[node])
        n_pre_neb_q_freq[node] = [0]*len(pre_neb_q[node])

        click_neb_q[node] = list(click_neb_q[node])
        top_neb_q[node] = list(top_neb_q[node])
        back_neb_q[node] = list(back_neb_q[node])
        pre_neb_q[node] = list(pre_neb_q[node])

        # mom = sum(click_neb_q[node])
        for i, nei in enumerate(click_neb_q[node]):
            n_click_neb_q_freq[node][i] = click_neb_q_freq[node][nei] #/ mom
            #n_click_neb_q[node][i] = nei
        # mom = sum(top_neb_q[node])
        for i, nei in enumerate(top_neb_q[node]):
            n_top_neb_q_freq[node][i] = top_neb_q_freq[node][nei] # mom
            #n_top_neb_q[node][i] = nei
        # mom = sum(back_neb_q[node])
        for i, nei in enumerate(back_neb_q[node]):
            n_back_neb_q_freq[node][i] = back_neb_q_freq[node][nei] #/ mom
            #n_back_neb_q[node][i] = nei
        # mom = sum(pre_neb_q[node])
        for i, nei in enumerate(pre_neb_q[node]):
            n_pre_neb_q_freq[node][i] = pre_neb_q_freq[node][nei] #/ mom
            #n_pre_neb_q[node][i] = nei

    for node in tqdm(click_neb_d):
        n_click_neb_d_freq[node] = [0]*len(click_neb_d[node])
        n_top_neb_d_freq[node] = [0]*len(top_neb_d[node])
        n_back_neb_d_freq[node] = [0]*len(back_neb_d[node])
        n_pre_neb_d_freq[node] = [0]*len(pre_neb_d[node])

        click_neb_d[node] = list(click_neb_d[node])
        top_neb_d[node] = list(top_neb_d[node])
        back_neb_d[node] = list(back_neb_d[node])
        pre_neb_d[node] = list(pre_neb_d[node])

        # mom = sum(click_neb_d[node])
        for i, nei in enumerate(click_neb_d[node]):
            n_click_neb_d_freq[node][i] = click_neb_d_freq[node][nei]# / mom
            #n_click_neb_d[node][i] = nei
        # mom = sum(top_neb_d[node])
        for i, nei in enumerate(top_neb_d[node]):
            n_top_neb_d_freq[node][i] = top_neb_d_freq[node][nei]# / mom
            #n_top_neb_d[node][i] = nei
        # mom = sum(back_neb_d[node])
        for i, nei in enumerate(back_neb_d[node]):
            n_back_neb_d_freq[node][i] = back_neb_d_freq[node][nei]# / mom
            #n_back_neb_d[node][i] = nei
        # mom = sum(pre_neb_d[node])
        for i, nei in enumerate(pre_neb_d[node]):
            n_pre_neb_d_freq[node][i] = pre_neb_d_freq[node][nei]# / mom
            #n_pre_neb_d[node][i] = nei
    print('End to get_edge_freq')


def cutting(node, new_freqs, neighs):
    deal = np.sum(new_freqs)
    cut = 0
    while deal > 100:
        norm = (1 / (new_freqs+1e-5)) * (new_freqs != 0)
        norm = norm / np.sum(norm)
        select = np.random.choice(neighs, 1, p=norm)
        new_freqs[neighs.index(select)] -= 1
        deal = np.sum(new_freqs)
        cut += 1
    #if cut > 0:
    #    print('node : {} cut {}'.format(node, cut))
    return new_freqs
    
    
def padding(new_freqs, neighs, max_):
    new_sum = np.sum(np.array(new_freqs))
    norm_freq = new_freqs / new_sum
    choices = np.random.choice(neighs, int(max_-new_sum), p=norm_freq)
    for c in choices:
        index = neighs.index(c)
        new_freqs[index] += 1
    return new_freqs, neighs

def get_deal(node, freqs, neigs):
    global max_freq
    min_ = 1
    max_ = 100
    max_freq = max(max_freq, max(freqs))
    new_freqs = freqs
    new_sum = np.sum(new_freqs)

    new_new_freqs = np.round(new_freqs / new_sum * max_, 0)
    deal = np.sum(new_new_freqs)

    if deal == 0:
        ind = np.array(list(reversed(np.argsort(freqs)))[:100], dtype=np.int)
        new_freqs = np.array(freqs)[ind].tolist()
        neigs = np.array(neigs)[ind].tolist()
        new_sum = np.sum(new_freqs)
        new_new_freqs = np.round(new_freqs / new_sum * max_, 0)
        deal = np.sum(new_new_freqs)

        # print('new_freqs = {} newnewfreqs = {} deal = {}'.format(new_freqs, new_new_freqs, deal))
        # print('edge = {} node = {} | neighs = {}'.format(e, get_str(node), [get_str(nei) for nei in neigs]))

    cnt = 0
    last_deal = deal
    while deal < 0.95 * max_:
        cnt += 1
        # try:
        new_new_freqs = np.round(new_new_freqs / deal * max_, 0)
        # except:
        #     print('new_new_freqs = {} deal = {}'.format(new_new_freqs, deal))
        deal = np.sum(new_new_freqs)
        # if deal == 0 :
        #     print('last deal = {} deal = {} new_new_freqs = {} freqs = {}'.format(last_deal, deal, new_new_freqs, freqs))
        #     assert 1==0
        if deal == last_deal: break

        last_deal = deal

    if deal > 100:
        new_new_freqs = cutting(node, new_new_freqs, neigs)
        deal = np.sum(new_new_freqs)

    flags = np.array(new_new_freqs != 0)
    new_neigs = (np.array(neigs)[flags]).tolist()
    n3_freq = (np.array(new_new_freqs)[flags]).tolist()
    if deal > 100:
        print('big deal = ', deal)
    if deal < max_:
        n4_freq, n4_neigs = padding(n3_freq, new_neigs, max_)
        deal = np.sum(n4_freq)
    else:
        n4_freq = n3_freq
        n4_neigs = new_neigs
    if deal != max_:
        print(
            'min deal = {}, n4 freq = {} n3_freq = {} new new freqs = {}'.format(deal, n4_freq, n3_freq, new_new_freqs))
    assert deal == max_
    if len(np.array(n4_freq).shape) > 1:
        print('node {} freqs {} new freqs {} n4 freq = {}'.format(node, freqs, new_neigs, n4_freq))
    return n4_freq, n4_neigs



def save_data():
    neighbor_dictlist = [
        click_neb_q,
        click_neb_d,
        top_neb_q,
        top_neb_d,
        back_neb_q,
        pre_neb_q,
        back_neb_d,
        pre_neb_d
    ]

    neighbor_freq_dictlist = [
        n_click_neb_q_freq,
        n_click_neb_d_freq,
        n_top_neb_q_freq,
        n_top_neb_d_freq,
        n_back_neb_q_freq,
        n_pre_neb_q_freq,
        n_back_neb_d_freq,
        n_pre_neb_d_freq
    ]
    
    '''
    get round
    '''
    print('--------------------BEGIN TO GET ROUND--------------------')
    new_neighbors = copy.deepcopy(neighbor_dictlist)
    new_freqs = copy.deepcopy(neighbor_freq_dictlist)
    
    deals = []
    for e in range(len(neighbor_freq_dictlist)):
        e_freq = neighbor_freq_dictlist[e]
        e_neigh = neighbor_dictlist[e]
        for node in e_freq:
            if len(e_freq[node]) > 0:
                new_f, new_n = get_deal(node, e_freq[node], e_neigh[node])
                new_freqs[e][node] = new_f
                new_neighbors[e][node] = new_n
                deals.append(np.sum(new_f))
          
    deals = sorted(deals)
    print('MAX FREQ = ', max_freq)
    print(deals[-50:])
    print('min deal = {}, max deal = {}'.format(deals[0], deals[-1]))
    '''
    get array
    '''
    print('--------------------BEGIN TO GET ARRAY--------------------')
    maxid = max(max(list(query2id.values())), max(list(doc2id.values()))) + 1
    minid = min(min(list(query2id.values())), min(list(doc2id.values())))

    max_nei_len = 0
    for e_nei in new_neighbors:
        for node in e_nei:
            max_nei_len = max(max_nei_len, len(e_nei[node]))

    neighbor = np.array([np.zeros([maxid, max_nei_len]) for i in range(8)])
    freq = np.array([np.zeros([maxid, max_nei_len]) for i in range(8)])
    for e in range(8):
        e_neighbor = new_neighbors[e]
        e_freqs = new_freqs[e]
        for node in tqdm(e_neighbor):
            length = len(e_neighbor[node])
            neighbor[e][node][:length] = e_neighbor[node]
            freq[e][node][:length] = np.array(e_freqs[node]).astype(np.int32)
    
    '''
    get prefix
    '''
    print('--------------------BEGIN TO GET PREFIX--------------------')
    print('neighbor shape = {} freq shape = {}'.format(neighbor.shape, freq.shape))
    node_cnt = neighbor.shape[1]
    edge_cnt = 8
    nei_cnt = neighbor.shape[2]

    prefix = np.zeros([edge_cnt, node_cnt, 100])
    pre_cnt = np.zeros([edge_cnt, node_cnt])

    for i in range(edge_cnt):
        for j in tqdm(range(node_cnt)):
            pre = 0
            if np.sum(freq[i][j]) > 0:
                for k in range(nei_cnt):
                    nei = neighbor[i][j][k]
                    length = freq[i][j][k]
                    # print('nei = {} length = {} pre = {} pre+len = {}'.format(nei, length, pre, pre+length))
                    prefix[i][j][int(pre):int(pre+length)] = nei
                    pre += length
                    if pre == 100:break
            if pre != 100 and pre != 0:
                print('pre =', pre)

        pickle.dump(prefix[i,:,:], open(r'./process_data_aol/sample_neighbor_prefix_1b{}.pkl'.format(i), 'wb'))
        
    print('--------------------SAVE OVER--------------------')

construct_ind(r'../../coca_data/aol/train.json', train=True)
construct_ind(r'../../coca_data/aol/test.json', train=False)
construct_ind(r'../../coca_data/aol/dev.json', train=False)
get_edge_freq()
save_data()

