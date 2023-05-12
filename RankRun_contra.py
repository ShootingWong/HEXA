import argparse
import random
import numpy as np
import torch
import torch.nn.utils as utils
from torch.utils.data import DataLoader
from transformers import AdamW, get_linear_schedule_with_warmup, BertTokenizer, BertModel
from Trec_Metrics import Metrics
from RankDataset_contra import RankDataset_contra, RankDataset_contra_test
from tqdm import tqdm
import os
import os.path as osp

from RankModel_pretower_contra import Ranker_nobert

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
parser.add_argument("--learning_rate",
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
                    default="/home/shuting_wang/bert-base-chinese",
                    type=str,
                    help="The path to save log.")
parser.add_argument("--pretrain_model_path",
                    default="",
                    type=str,
                    help="The path to save log.")
parser.add_argument("--log_path",
                    default="./log/",
                    type=str,
                    help="The path to save log.")
parser.add_argument("--devices",
                    default="1",
                    type=str,
                    help="The gpu devices can be seen by the process")
parser.add_argument("--model_type",
                    default="Ranker",
                    type=str,
                    help="The gpu devices can be seen by the process")
                 
parser.add_argument("--suffix",
                    default="session.frezemb.shartype",
                    type=str,
                    help="The gpu devices can be seen by the process")   
                    
parser.add_argument("--performs",
                    default="0,0,0,0,0,0",
                    type=str,
                    help="The gpu devices can be seen by the process")   



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
args.log_path += args.model_type + "." + args.task + ".log" + "." + args.suffix + '.train' if args.is_training == 1 else '.test'
score_file_prefix = result_path + args.model_type + "." + args.task
args.score_file_path = score_file_prefix + "." + args.score_file_path + "." + args.suffix
args.score_file_pre_path = score_file_prefix + "." + args.score_file_pre_path + "." + args.suffix
args.is_training = bool(args.is_training)


logger = open(args.log_path, "a")
device = torch.device("cuda:0")
print(args)
logger.write("\nHyper-parameters:\n")
args_dict = vars(args)

for k, v in args_dict.items():
    logger.write(str(k) + "\t" + str(v) + "\n")

if args.task == "aol":
    train_data = "./aol/train.rank.onlypos.txt"
    test_data = "./aol/test.rank.small.txt"
    predict_data = "./aol/test.rank.txt"
    tokenizer = BertTokenizer.from_pretrained(args.bert_model_path)
    additional_tokens = 3
    tokenizer.add_tokens("[eos]")
    tokenizer.add_tokens("[term_del]")
    tokenizer.add_tokens("[sent_del]")
    EOS = tokenizer.convert_tokens_to_ids("[eos]")
    print('EOS = ', EOS)
elif args.task == "tiangong":
    train_data = "./tiangong/train.onlypos.txt"
    test_data = "./tiangong/test_last.txt"
    predict_last_data = "./tiangong/test_last.txt"
    tokenizer = BertTokenizer.from_pretrained(args.bert_model_path)
    additional_tokens = 4
    tokenizer.add_tokens("[eos]")
    tokenizer.add_tokens("[empty_d]")
    tokenizer.add_tokens("[term_del]")
    tokenizer.add_tokens("[sent_del]")
else:
    assert False


def set_seed(seed=0):
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
    model = Ranker_nobert(additional_tokens, config_state)

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

def train_step(model, train_data, bce_loss):
    with torch.no_grad():
        for key in train_data.keys():
            train_data[key] = train_data[key].cuda()

    loss = model.forward(train_data)

    return loss

def fit(model, X_train, X_test):
    print('BEGIN TO FIT!')
    train_dataset = RankDataset_contra(X_train, 128, 128, tokenizer)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    print('PREPARE DATASTE OVER!')

    optimizer = AdamW(model.parameters(), lr=args.learning_rate)
    t_total = int(len(train_dataset) * args.epochs // args.batch_size)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(t_total) * 0,
                                                num_training_steps=t_total)
    one_epoch_step = len(train_dataset) // args.batch_size
    bce_loss = torch.nn.BCEWithLogitsLoss()
    best_result = [float(v) for v in args.performs.split(',')]#[0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    print('Init best result = ', best_result)

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
            utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            scheduler.step()
            model.zero_grad()
            
            for param_group in optimizer.param_groups:
                args.learning_rate = param_group['lr']
                lr = param_group['lr']
            epoch_iterator.set_postfix(lr=lr, loss=loss.detach().cpu().numpy())
            if i > 0 and i % (one_epoch_step // 5) == 0:
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
    if args.task == 'tiangong' and 'last' in test_data :bg = 2

    if not is_test and sum(result[bg:]) > sum(best_result[bg:]):
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
        if args.task == "tiangong":
            print(
                "Previsou Query Best Result: MAP: %.4f MRR: %.4f NDCG@1: %.4f NDCG@3: %.4f NDCG@5: %.4f NDCG@10: %.4f" % (
                result_pre[0], result_pre[1], result_pre[2], result_pre[3], result_pre[4], result_pre[5]))
    return best_result


def predict(model, X_test):
    model.eval()
    print('X_test = ', X_test)
    test_dataset = RankDataset_contra_test(X_test, 128, 128, tokenizer)
    test_dataloader = DataLoader(test_dataset, batch_size=args.test_batch_size, shuffle=False, num_workers=8)
    y_pred = []
    y_label = []
    with torch.no_grad():
        epoch_iterator = tqdm(test_dataloader, ncols=120, leave=False)
        for i, test_data in enumerate(epoch_iterator):
            with torch.no_grad():
                for key in test_data.keys():
                    test_data[key] = test_data[key].cuda()
            y_pred_test = model.forward(test_data, is_test=True)
            y_pred.append(y_pred_test.data.cpu().numpy().reshape(-1))
            y_tmp_label = test_data["label"].data.cpu().numpy().reshape(-1)
            y_label.append(y_tmp_label)
    y_pred = np.concatenate(y_pred, axis=0).tolist()
    y_label = np.concatenate(y_label, axis=0).tolist()

    return y_pred, y_label


def test_model(config_state, train=True):
    
    bert_model = BertModel.from_pretrained(args.bert_model_path)
    bert_model.resize_token_embeddings(bert_model.config.vocab_size + additional_tokens)
    
    model = Ranker_nobert(additional_tokens, config_state)
    
    model_state_dict = torch.load(args.save_path)
    model.load_state_dict({k.replace('module.', ''): v for k, v in model_state_dict.items()})
    model = model.cuda()
    model = torch.nn.DataParallel(model)
    if args.task == "aol":
        evaluate(model, predict_data, None, [0.0, 0.0, 0.0, 0.0, 0.0, 0.0], is_test=True)
    elif args.task == "tiangong":
        evaluate(model, predict_last_data, None, [0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 
                 is_test=True)


if __name__ == '__main__':
    set_seed()
    config_state = eval(args.config_type)()
    if args.is_training:
        train_model(config_state)
        print("start test...")
        test_model(config_state)
    else:
        print('ONLY TEST')
        test_model(config_state, train=False)