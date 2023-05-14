import argparse
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.utils as utils
from torch.utils.data import DataLoader
from transformers import AdamW, get_linear_schedule_with_warmup, BertTokenizer, BertModel
from Trec_Metrics import Metrics
from RankDataset import RankDataset_point, RankDataset_test
from tqdm import tqdm
import os
import os.path as osp
from time import time
from RankModel import Ranker, BertRanker, Ranker_emb
from config import *

import pynvml

parser = argparse.ArgumentParser()

parser.add_argument("--config_type",
                    default="basic_config",
                    type=str,
                    help="The type of config")
parser.add_argument("--is_training",
                    default=1,
                    type=int,
                    help="Training model or evaluating model?")
parser.add_argument("--per_gpu_batch_size",
                    default=16,
                    type=int,
                    help="The batch size.")
parser.add_argument("--per_gpu_test_batch_size",
                    default=160,
                    type=int,
                    help="The batch size.")
parser.add_argument("--test_emb",
                    default=0,
                    type=int,
                    help="The batch size.")
parser.add_argument("--heter",
                    default=1, 
                    type=int,
                    help="The batch size.")
parser.add_argument("--learning_rate",
                    default=1e-5,
                    type=float,
                    help="The initial learning rate for Adam.")
parser.add_argument("--bert_lr",
                    default=1e-5,
                    type=float,
                    help="The initial learning rate for Adam.")
parser.add_argument("--task",
                    default="aol",
                    type=str,
                    help="Task")
parser.add_argument("--epochs",
                    default=3,
                    type=int,
                    help="Total number of training epochs to perform.")
parser.add_argument("--emb_path",
                    default="",
                    type=str,
                    help="The path to save model.")
parser.add_argument("--save_path",
                    default="./model/",
                    type=str,
                    help="The path to save model.")
parser.add_argument("--score_file_path",
                    default="score_file.txt",
                    type=str,
                    help="The path to save model.")
parser.add_argument("--score_file_pre_path",
                    default="score_file.preq.txt",
                    type=str,
                    help="The path to save model.")
parser.add_argument("--model_path",
                    default="",
                    type=str,
                    help="The path to save log.")
parser.add_argument("--bert_model_path",
                    default="./bert",
                    type=str,
                    help="The path of pretrained bert path.")
parser.add_argument("--log_path",
                    default="./log/",
                    type=str,
                    help="The path to save log.")
parser.add_argument("--devices",
                    default="1",
                    type=str,
                    help="The gpu devices can be seen by the process")
parser.add_argument("--seed",
                    default="0",
                    type=str,
                    help="The seed used to fix the initialization of the model")
parser.add_argument("--model_type",
                    default="Ranker",
                    type=str,
                    help="")
parser.add_argument("--test_type",
                    default="last",
                    type=str,
                    help="")
parser.add_argument("--loss_type",
                    default="point",
                    type=str,
                    help="")
                 
parser.add_argument("--suffix",
                    default="",
                    type=str,
                    help="The suffix to figure out the different training logs")   


class PairwiseLoss(nn.Module):
    def __init__(self):
        super(PairwiseLoss, self).__init__()

    def forward(self, score_1, score_2):
        pij = 1 / (torch.exp(score_2 - score_1) + 1)
        pji = 1 / (torch.exp(score_1 - score_2) + 1)
        return ((-torch.log(torch.softmax(torch.cat([pij, pji], dim=1), dim=1))[:, 0])).mean(0)

def see_gpu():
    pynvml.nvmlInit()
    
    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
    meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
    return meminfo.used/1024/1024
    
def mkdir_if_missing(directory):
    if not osp.exists(directory):
        try:
            os.makedirs(directory)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise
                
args = parser.parse_args()

mkdir_if_missing('./output')
mkdir_if_missing(args.save_path)
mkdir_if_missing(args.log_path)
args.batch_size = args.per_gpu_batch_size #* torch.cuda.device_count()
args.test_batch_size = args.per_gpu_test_batch_size #* torch.cuda.device_count()
result_path = "./output/" + args.task + "/"
args.save_path += args.model_type + "." + args.task + "." + args.suffix
args.log_path += args.model_type + "." + args.task + ".log" + "." + args.suffix
score_file_prefix = result_path + args.model_type + "." + args.task
args.score_file_path = score_file_prefix + "." + args.score_file_path + "." + args.suffix
args.score_file_pre_path = score_file_prefix + "." + args.score_file_pre_path + "." + args.suffix
args.is_training = bool(args.is_training)
args.heter = bool(args.heter)

logger = open(args.log_path, "a")
device = torch.device("cuda:0")
print(args)
logger.write("\nHyper-parameters:\n")
args_dict = vars(args)

for k, v in args_dict.items():
    logger.write(str(k) + "\t" + str(v) + "\n")

if args.task == "aol":
    train_data = "./aol/train.rank.txt"
    test_data = "./aol/dev.rank.txt"
    predict_data = "./aol/test.rank.txt"
    tokenizer = BertTokenizer.from_pretrained(args.bert_model_path)
    additional_tokens = 3
    tokenizer.add_tokens("[eos]")
    tokenizer.add_tokens("[term_del]")
    tokenizer.add_tokens("[sent_del]")
    EOS = tokenizer.convert_tokens_to_ids("[eos]")
    print('EOS = ', EOS)
elif args.task == "tiangong":
    train_data = "./tiangong/train.txt"
    predict_last_data = "./tiangong/dev_last.txt"
    test_data = predict_last_data
    tokenizer = BertTokenizer.from_pretrained(args.bert_model_path)
    additional_tokens = 4
    
    tokenizer.add_tokens("[eos]")
    tokenizer.add_tokens("[empty_d]")
    tokenizer.add_tokens("[term_del]")
    tokenizer.add_tokens("[sent_del]")#inline with the RankContra

else:
    assert False


def set_seed(seed=666):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def train_model(config_state):
    bert_model = BertModel.from_pretrained(args.bert_model_path)
    bert_model.resize_token_embeddings(bert_model.config.vocab_size + additional_tokens)

    print('PREPARE BERT MODEL OVER!')
    if args.model_type == 'Ranker':
        model = Ranker(bert_model, config_state, args.heter, weight=None)
        if args.task == 'tiangong':
            for name, param in model.named_parameters():
                print(name, param.requires_grad)
    else:
        model = BertRanker(bert_model, config_state)

    
    if args.emb_path != '':
        graph_embed = Ranker_nobert(bert_model, config_state)
        graph_embed.load_state_dict(torch.load(args.emb_path))
        print('Successfully load state from ', args.emb_path)
        model.graph_ember.load_state_dict(graph_embed.graph_ember.state_dict())
        model.classifier_graph.load_state_dict(graph_embed.classifier.state_dict())
        del graph_embed
    if args.model_path != '':
        model.load_state_dict(torch.load(args.model_path))
        
    print('torch.cuda.device_count() = ',torch.cuda.device_count())
    if torch.cuda.device_count() > 1:
        print('PARALLEL MODEL')
        model = torch.nn.DataParallel(model, device_ids=[0])
    model = model.to(device)
    
    print('model cost ', see_gpu())
    print('PREPARE WHOLE MODEL OVER!')
    fit(model, train_data, test_data)


def train_step(model, train_data, loss_fun):
    with torch.no_grad():
        for key in train_data.keys():
            train_data[key] = train_data[key].cuda()
            
    if args.loss_type == 'point':
        y_pred = model.forward(**train_data)
        batch_y = train_data["label"].float()
        loss = loss_fun(y_pred, batch_y)
    else:
        y_pred_p = model.forward(train_data['input_ids_p'], train_data['token_type_ids_p'], train_data['attention_mask_p'], train_data['qid'], train_data['did_p'], train_data['session_qid'], train_data['session_did'], train_data['session_len'])
        y_pred_n = model.forward(train_data['input_ids_n'], train_data['token_type_ids_n'], train_data['attention_mask_n'], train_data['qid'], train_data['did_n'], train_data['session_qid'], train_data['session_did'], train_data['session_len'])
        
        loss = loss_fun(y_pred_p, y_pred_n)
        
    return loss


def fit(model, X_train, X_test):
    print('BEGIN TO FIT!')
    train_dataset = RankDataset_point(X_train, 128, 10, tokenizer)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    print('PREPARE DATASTE OVER!')
    
    
    optimizer_grouped_parameters = [ \
        {'params': [p for n, p in model.named_parameters() if 'bert' in n], 'lr': args.bert_lr}, \
        {'params': [p for n, p in model.named_parameters() if 'bert' not in n]} \
        ]
    

    optimizer = AdamW(model.parameters(), lr=args.learning_rate)
    t_total = int(len(train_dataset) * args.epochs // args.batch_size)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(t_total) * 0,
                                                num_training_steps=t_total)
    one_epoch_step = len(train_dataset) // args.batch_size
    if args.loss_type == 'point':
        bce_loss = torch.nn.BCEWithLogitsLoss()
    else:
        bce_loss = PairwiseLoss()
    best_result = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

    for epoch in range(args.epochs):
        print("\nEpoch ", epoch + 1, "/", args.epochs)
        logger.write("Epoch " + str(epoch + 1) + "/" + str(args.epochs) + "\n")
        avg_loss = 0
        model.train()
        epoch_iterator = tqdm(train_dataloader)
        for i, training_data in enumerate(epoch_iterator):
            loss = train_step(model, training_data, bce_loss)
            loss = loss.mean()
            loss.backward()
            utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()
            scheduler.step()
            model.zero_grad()
            
            for param_group in optimizer.param_groups:
                args.learning_rate = param_group['lr']
                lr = param_group['lr']
            epoch_iterator.set_postfix(lr=lr, loss=loss.detach().cpu().numpy())
            if i > 0 and i % (one_epoch_step // 10) == 0:
                best_result = evaluate(model, X_test, bce_loss, best_result)
                model.train()
            avg_loss += loss.item()

        cnt = len(train_dataset) // args.batch_size + 1
        tqdm.write("EPOCH {}: Average loss:{:.6f} ".format(epoch, avg_loss / cnt))
        best_result = evaluate(model, X_test, bce_loss, best_result)
    logger.close()

def evaluate(model, X_test, bce_loss, best_result, is_test=False):
    if args.task == "aol":
        y_pred, y_label = predict(model, X_test)
        metrics = Metrics(args.score_file_path, segment=50)
    elif args.task == "tiangong":
        y_pred, y_label = predict(model, X_test)
        metrics = Metrics(args.score_file_path, segment=10)

    with open(args.score_file_path, 'w') as output:
        for score, label in zip(y_pred, y_label):
            output.write(str(score) + '\t' + str(label) + '\n')

    result = metrics.evaluate_all_metrics()
    
    bg = 0
    ed = 6
    if args.test_type == 'last':bg = 2

    if not is_test and sum(result[bg:ed]) > sum(best_result[bg:ed]):
        best_result = result
        print("Best Result: MAP: %.4f MRR: %.4f NDCG@1: %.4f NDCG@3: %.4f NDCG@5: %.4f NDCG@10: %.4f" % (
        best_result[0], best_result[1], best_result[2], best_result[3], best_result[4], best_result[5]))
        logger.write("Best Result: MAP: %.4f MRR: %.4f NDCG@1: %.4f NDCG@3: %.4f NDCG@5: %.4f NDCG@10: %.4f \n" % (
        best_result[0], best_result[1], best_result[2], best_result[3], best_result[4], best_result[5]))
        logger.flush()
        model_to_save = model.module if hasattr(model, 'module') else model
        torch.save(model_to_save.state_dict(), args.save_path)
    if is_test:
        print("Best Result: MAP: %.4f MRR: %.4f NDCG@1: %.4f NDCG@3: %.4f NDCG@5: %.4f NDCG@10: %.4f" % (
        result[0], result[1], result[2], result[3], result[4], result[5]))
    return best_result


def predict(model, X_test):
    model.eval()
    candi_cnt = 50 if args.task == 'aol' else 10
    test_dataset = RankDataset_test(candi_cnt, X_test, 128, 10, tokenizer)
    test_dataloader = DataLoader(test_dataset, batch_size=args.test_batch_size, shuffle=False, num_workers=4)
    print('All last test query is ', len(test_dataset))
    y_pred = []
    y_label = []
    with torch.no_grad():
        epoch_iterator = tqdm(test_dataloader, ncols=120, leave=False)
        time1 = time()
        for i, test_data in enumerate(epoch_iterator):
            with torch.no_grad():
                for key in test_data.keys():
                    test_data[key] = test_data[key].cuda()
            y_pred_test = model.test(**test_data)
            y_pred.append(y_pred_test.data.cpu().numpy().reshape(-1))
            y_tmp_label = test_data["label"].data.cpu().numpy().reshape(-1)
            y_label.append(y_tmp_label)
        time2 = time()
        print('PRED COST ', time2-time1)
    y_pred = np.concatenate(y_pred, axis=0).tolist()
    y_label = np.concatenate(y_label, axis=0).tolist()
    
    return y_pred, y_label

def test_model(config_state, train=True):
    
    bert_model = BertModel.from_pretrained(args.bert_model_path)
    bert_model.resize_token_embeddings(bert_model.config.vocab_size + additional_tokens)
    if args.test_emb == 1:
        print('Ranker_emb')
        model = Ranker_emb(config_state)
    else:
        if args.model_type == 'Ranker':
            model = Ranker(bert_model,config_state, args.heter)
        else:
            model = BertRanker(bert_model, config_state)
    if args.test_emb == 0:
        
        model_state_dict = torch.load(args.save_path)
        model.load_state_dict({k.replace('module.', ''): v for k, v in model_state_dict.items()}, strict=False)
        print('Load model state dict from ', args.save_path)

    model = model.cuda()
    #model = torch.nn.DataParallel(model)
    if args.task == "aol":
        evaluate(model, predict_data, None, [0.0, 0.0, 0.0, 0.0, 0.0, 0.0], is_test=True)
    elif args.task == "tiangong":
        evaluate(model, predict_last_data, None, [0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 
                 is_test=True)


if __name__ == '__main__':
    set_seed(int(args.seed))
    config_state = eval(args.config_type)()
    if args.is_training:
        train_model(config_state)
        print("start test...")
        test_model(config_state)
    else:
        print('ONLY TEST')
        test_model(config_state, train=False)
