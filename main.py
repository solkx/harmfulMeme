import argparse
import random
import numpy as np
import prettytable as pt
import torch
import torch.autograd
import torch.nn as nn
import torch.optim as optim
import transformers
from torch.utils.data import DataLoader
from tqdm import tqdm
import config
import data_loader
import utils
from model import Model
import os
import torch.nn.functional as F

class Trainer(object):
    def __init__(self, model):
        self.model = model
        self.criterion = nn.CrossEntropyLoss()

        bert_params = set(self.model.bert.parameters())
        vit_params = set(self.model.vit.parameters())
        other_params = list(set(self.model.parameters()) - bert_params - vit_params)
        # other_params = list(set(self.model.parameters()) - bert_params)
        no_decay = ['bias', 'LayerNorm.weight']
        params = [
            {'params': [p for n, p in model.bert.named_parameters() if not any(nd in n for nd in no_decay)],
                'lr': config.bert_learning_rate,
                'weight_decay': config.weight_decay},
            {'params': [p for n, p in model.bert.named_parameters() if any(nd in n for nd in no_decay)],
                'lr': config.bert_learning_rate,
                'weight_decay': 0.0},
            {'params': [p for n, p in model.vit.named_parameters() if not any(nd in n for nd in no_decay)],
             'lr': config.bert_learning_rate,
             'weight_decay': config.weight_decay},
            {'params': other_params,
                'lr': config.learning_rate,
                'weight_decay': config.weight_decay},
        ]

        self.optimizer = optim.AdamW(params, lr=config.learning_rate, weight_decay=config.weight_decay)
        self.scheduler = transformers.get_linear_schedule_with_warmup(self.optimizer,
                                                                      num_warmup_steps=config.warm_factor * updates_total,
                                                                      num_training_steps=updates_total)

    def calu_res(self, p, r, c_p, c_r=None):
        if not c_r:
            c_r = c_p
        try:
            precious = c_p / p
        except ZeroDivisionError:
            precious = 0
        try:
            recall = c_r / r
        except ZeroDivisionError:
            recall = 0

        try:
            f1 = 2 * precious * recall / (precious + recall)
        except ZeroDivisionError:
            f1 = 0
        return round(precious, 4), round(recall, 4), round(f1, 4)
        
    def train(self, epoch, data_loader):
        self.model.train()
        loss_list = []
        for i, data_batch in enumerate(tqdm(data_loader)):
            sents = data_batch[-1]
            data_batch = [data.to(device) for data in data_batch[:-1]]
            bert_inputs, bert_inputs_token, bert_inputs_mask, images, pieces2word, labels, sent_l, sememes, adjs = data_batch
            
            loss = model(bert_inputs_token, images, pieces2word, labels, sent_l, sememes, adjs, sents)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5.0)
            self.optimizer.step()
            self.optimizer.zero_grad()

            loss_list.append(loss.cpu().item())

            self.scheduler.step()

    
        table = pt.PrettyTable(["Train {}".format(epoch), "Loss"])
        table.add_row(["", "{:.4f}".format(np.mean(loss_list))])
        logger.info("\n{}".format(table))

        return loss

    def eval(self, epoch, data_loader, config, is_test=False):
        self.model.eval()
        label_num = len(config.vocab.label2id[config.is_b])
        total_result = {i:[0, 0, 0] for i in range(label_num)}
        
        with torch.no_grad():
            for i, data_batch in enumerate(tqdm(data_loader)):
                sents = data_batch[-1]
                data_batch = [data.to(device) for data in data_batch[:-1]]

                bert_inputs, bert_inputs_token, bert_inputs_mask, images, pieces2word, labels, sent_l, sememes, adjs = data_batch
                
                result = model(bert_inputs_token, images, pieces2word, labels, sent_l, sememes, adjs, sents, is_train=False)
                for label, prc in result.items():
                    total_result[label][0] += prc[0]
                    total_result[label][1] += prc[1]
                    total_result[label][2] += prc[2]
        
        total_result = {k:self.calu_res(v[0], v[1], v[2]) for k, v in total_result.items()}
        mirco_p, mirco_r, mirco_f1 = 0, 0, 0       

        Title = "DEV" if not is_test else "TEST"
        
        table = pt.PrettyTable([f"Gold {Title} {epoch}", "Precision", "Recall", "F1"])
        for label_id, res in total_result.items():
            if label_id == -1:
                continue
            mirco_p += res[0] 
            mirco_r += res[1] 
            mirco_f1 += res[2]
            table.add_row([config.vocab.id2label[config.is_b][label_id]] + ["{:3.4f}".format(x) for x in [res[0], res[1], res[2]]])
        table.add_row(["all"] + ["{:3.4f}".format(x) for x in [mirco_p/label_num, mirco_r/label_num, mirco_f1/label_num]])

        # for label_id, res in total_result.items():
        #     if label_id > 0:
        #         continue
        #     mirco_p += res[0] 
        #     mirco_r += res[1] 
        #     mirco_f1 += res[2]
        #     if label_id == -1:
        #         table.add_row(["harmful"] + ["{:3.4f}".format(x) for x in [res[0], res[1], res[2]]])
        #     else:
        #         table.add_row([config.vocab.id2label[config.is_b][label_id]] + ["{:3.4f}".format(x) for x in [res[0], res[1], res[2]]])
        # table.add_row(["all"] + ["{:3.4f}".format(x) for x in [mirco_p/label_num, mirco_r/label_num, mirco_f1/label_num]])


        logger.info("\n{}".format(table))
        return np.mean(mirco_f1/label_num)
    

    def save(self, path):
        torch.save(self.model.state_dict(), path)

    def load(self, path):
        self.model.load_state_dict(torch.load(path))

def seed_torch(seed=3306):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed) # 为了禁止hash随机化，使得实验可复现
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='./config.json')
    parser.add_argument('--gpu_id', type=int)

    parser.add_argument('--epochs', type=int)
    parser.add_argument('--batch_size', type=int)

    parser.add_argument('--learning_rate', type=float)
    parser.add_argument('--weight_decay', type=float)

    parser.add_argument('--bert_name', type=str)
    parser.add_argument('--bert_learning_rate', type=float)
    parser.add_argument('--warm_factor', type=float)
    parser.add_argument('--seed', type=int)
    parser.add_argument('--dropout', type=float)
    parser.add_argument('--logPath', type=str, default='log')
    parser.add_argument('--model_name', type=str, default='best_model.pth')
    parser.add_argument('--is_h', type=str2bool, default='True')
    parser.add_argument('--is_b', type=str2bool, default='False')
    parser.add_argument('--use_sememe', type=str2bool, default='True')

    args = parser.parse_args()
    
    config = config.Config(args)
    
    torch.autograd.set_detect_anomaly(True)
    seed_torch(config.seed)
    logger = utils.get_logger(config)
    logger.info(config)
    config.logger = logger

    if torch.cuda.is_available():
        device = f"cuda:{config.gpu_id}"
    else:
        device = "cpu"
        
    config.device = device



    logger.info("Loading Data")
    datasets = data_loader.load_data_bert(config)

    train_loader, test_loader = (
        DataLoader(dataset=dataset,
                   batch_size=config.batch_size,
                   collate_fn=data_loader.collate_fn,
                   shuffle=i == 0,
                   num_workers=4)
        for i, dataset in enumerate(datasets)
    )

    updates_total = len(datasets[0]) // config.batch_size * config.epochs
    logger.info("Building Model")
    model = Model(config)

    model = model.to(device)

    trainer = Trainer(model)
    best_f1, best_f1_gold = 0, 0
    for i in range(config.epochs):
        logger.info("Epoch: {}".format(i))
        trainer.train(i, train_loader)
        all_avg_f1_gold = trainer.eval(i, test_loader, config, is_test=True)
        if best_f1_gold <= all_avg_f1_gold:
            best_f1_gold = all_avg_f1_gold
            trainer.save(config.model_name)
    trainer.load(config.model_name)
    trainer.eval("Final", test_loader, config, True)
