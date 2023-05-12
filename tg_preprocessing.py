import json
from transformers import BertTokenizer, BertModel
from tqdm import tqdm
# import h5py
import numpy as np
import random
import math
import pickle

random.seed(0)

query2id = dict()
doc2id = dict()

query2id_path = r'./process_data_tg/query2id.pkl'
doc2id_path = r'./process_data_tg/doc2id.pkl'

def init_dict():
    global query2id, doc2id
    query2id = pickle.load(open(query2id_path, 'rb'))
    doc2id = pickle.load(open(doc2id_path, 'rb'))

def make_tiangong_train_datasest(fromfile, tofile, is_small=True):
    global query2id, doc2id
    QUERY_MAX_LEN = 10
    DOC_MAX_LEN = 25
    with open(fromfile, "r") as fr:
        with open(tofile, "w") as fw:
            lines = fr.readlines()
            print('line cnt = ', len(lines))
            if is_small:
                lines = random.sample(lines, 2000)
            for line in tqdm(lines):
                line = json.loads(line)
                history = ""
                history_id = ""
                for q in line["query"]:
                    has_click = False
                    query = " ".join(q["text"].split()[:QUERY_MAX_LEN])
                    history += query + "\t"
                    history_id += str(query2id[query]) + '\t'
                    for doc in q["clicks"]:
                        title = " ".join(doc["title"].split()[:DOC_MAX_LEN])
                        if title == "":
                            title = "[empty_d]"
                        if "label" in doc:
                            sample = "1" + "\t" + history + title + '\t' + history_id + str(doc2id[title]) + "\n"
                            fw.write(sample)
                            if not has_click:
                                new_history = history + title + "\t"
                                new_history_id = history_id + str(doc2id[title]) + '\t'
                                has_click = True
                        else:
                            sample = "0" + "\t" + history + title + '\t' + history_id + str(doc2id[title]) + "\n"
                            fw.write(sample)
                    if has_click:
                        history = new_history
                        history_id = new_history_id
                    else:
                        history += "[empty_d]" + '\t'
                        history_id += str(doc2id["[empty_d]"]) + '\t'


def make_tiangong_test_lastq_datasest(fromfile, tofile, is_small=True):
    global query2id, doc2id
    QUERY_MAX_LEN = 10
    DOC_MAX_LEN = 25
    with open(fromfile, "r") as fr:
        with open(tofile, "w") as fw:
            lines = fr.readlines()
            print('line cnt = ', len(lines))
            if is_small:
                lines = random.sample(lines, 2000)
            for line in tqdm(lines):
                line = json.loads(line)
                history = ""
                history_id = ""
                for q_id, q in enumerate(line["query"]):
                    has_click = False
                    query = " ".join(q["text"].split()[:QUERY_MAX_LEN])
                    history += query + "\t"
                    history_id += str(query2id[query]) + '\t'
                    for doc in q["clicks"]:
                        title = " ".join(doc["title"].split()[:DOC_MAX_LEN])
                        if title == "":
                            title = "[empty_d]"
                        if q_id == len(line["query"]) - 1:
                            sample = doc["label"] + "\t" + history + title + '\t' + history_id + str(doc2id[title]) + "\n"
                            fw.write(sample)
                        else:
                            if "label" in doc:
                                if not has_click:
                                    new_history = history + title + "\t"
                                    new_history_id = history_id + str(doc2id[title]) + '\t'
                                    has_click = True

                    if has_click:
                        history = new_history
                        history_id = new_history_id
                    else:
                        history += "[empty_d]" + '\t'
                        history_id += str(doc2id["[empty_d]"]) + '\t'



def make_tiangong_test_preq_datasest(fromfile, tofile, is_small=True):
    global query2id, doc2id
    QUERY_MAX_LEN = 10
    DOC_MAX_LEN = 25
    with open(fromfile, "r") as fr:
        with open(tofile, "w") as fw:
            lines = fr.readlines()
            print('line cnt = ', len(lines))
            if is_small:
                lines = random.sample(lines, 2000)
            for line in tqdm(lines):
                line = json.loads(line)
                history = ""
                history_id = ""
                for q_id, q in enumerate(line["query"]):
                    samples = []
                    has_click = False
                    query = " ".join(q["text"].split()[:QUERY_MAX_LEN])
                    history += query + "\t"
                    history_id += str(query2id[query]) + '\t'
                    for doc in q["clicks"]:
                        title = " ".join(doc["title"].split()[:DOC_MAX_LEN])
                        if title == "":
                            title = "[empty_d]"
                        if q_id == len(line["query"]) - 1:
                            continue
                        if "label" in doc:
                            sample = "1" + "\t" + history + title + '\t' + history_id + str(doc2id[title]) + "\n"
                            samples.append(sample)
                            # fw.write(sample)
                            if not has_click:
                                new_history = history + title + "\t"
                                new_history_id = history_id + str(doc2id[title]) + '\t'
                                has_click = True
                        else:
                            sample = "0" + "\t" + history + title + '\t' + history_id + str(doc2id[title]) + "\n"
                            samples.append(sample)
                            # fw.write(sample)

                    if has_click:
                        history = new_history
                        history_id = new_history_id
                        for sample in samples:
                            fw.write(sample)
                    else:
                        history += "[empty_d]" + '\t'
                        history_id += str(doc2id["[empty_d]"]) + '\t'


init_dict()

train_file = '../coca_data/tiangong/train_candidate.json'
new_train_file = './tiangong/train.txt'
test_file = '../coca_data/tiangong/test_candidate.json'
new_dev_last = './tiangong/test_last.txt'


make_tiangong_train_datasest(train_file, new_train_file, is_small=False)
make_tiangong_test_lastq_datasest(test_file, new_dev_last, is_small=False)





