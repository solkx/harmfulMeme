import json
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
import numpy as np
import prettytable as pt
from gensim.models import KeyedVectors
from transformers import AutoTokenizer
import os
import utils
import requests
import cv2
from tqdm import tqdm
import OpenHowNet
import re

os.environ["TOKENIZERS_PARALLELISM"] = "false"

dis2idx = np.zeros((1000), dtype='int64')
dis2idx[1] = 1
dis2idx[2:] = 2
dis2idx[4:] = 3
dis2idx[8:] = 4
dis2idx[16:] = 5
dis2idx[32:] = 6
dis2idx[64:] = 7
dis2idx[128:] = 8
dis2idx[256:] = 9


class Vocabulary(object):
    PAD = '<pad>'
    UNK = '<unk>'
    SUC = '<suc>'
    PRE = '<pre>'
    NSUC = '<nsuc>'
    DSUC = '<dsuc>'
    CON = '<con>'



    def __init__(self, frequency=0):
        self.token2id = {self.PAD: 0, self.UNK: 1}
        self.id2token = {0: self.PAD, 1: self.UNK}
        self.token2count = {self.PAD: 1000, self.UNK: 1000}
        self.frequency = frequency

        self.char2id = {self.PAD: 0, self.UNK: 1}
        self.id2char = {0: self.PAD, 1: self.UNK}

        self.label2id = {False:{"Non-harmful": 0, "Targeted Harmful": 1, "Sexual Innuendo": 2, "General Offense": 3, "Dispirited Culture": 4}, True:{"Non-harmful": 0, "harmful": 1}}

        self.id2label = {False:{0: "Non-harmful", 1: "Targeted Harmful", 2: "Sexual Innuendo", 3: "General Offense", 4: "Dispirited Culture"},True:{0: "Non-harmful", 1: "harmful"}}

    def add_token(self, token):
        token = token.lower()
        if token in self.token2id:
            self.token2count[token] += 1
        else:
            self.token2id[token] = len(self.token2id)
            self.id2token[self.token2id[token]] = token
            self.token2count[token] = 1

        assert token == self.id2token[self.token2id[token]]

    def add_char(self, char):
        char = char.lower()
        if char not in self.char2id:
            self.char2id[char] = len(self.char2id)
            self.id2char[self.char2id[char]] = char

        assert char == self.id2char[self.char2id[char]]

    def add_label(self, label):
        label = label.lower()
        if label not in self.label2id:
            self.label2id[label] = len(self.label2id)
            self.id2label[self.label2id[label]] = label

        assert label == self.id2label[self.label2id[label]]

    def remove_low_frequency_token(self):
        new_token2id = {self.PAD: 0, self.UNK: 1}
        new_id2token = {0: self.PAD, 1: self.UNK}

        for token in self.token2id:
            if self.token2count[token] > self.frequency and token not in new_token2id:
                new_token2id[token] = len(new_token2id)
                new_id2token[new_token2id[token]] = token

        self.token2id = new_token2id
        self.id2token = new_id2token

    def __len__(self):
        return len(self.token2id)

    # def label_to_id(self, label):
    #     label = label.lower()
    #     return self.label2id[label]

    def encode(self, text):
        return [self.token2id.get(x.lower(), 1) for x in text]

    def encode_char(self, text):
        return [self.char2id.get(x, 1) for x in text]

    def decode(self, ids):
        return [self.id2token.get(x) for x in ids]


def pad_and_stack_tensors(tensor_group):
    # 找到这一组张量的 x_max
    x_max = max(tensor.shape[0] for tensor in tensor_group)
    
    # 创建一个列表，用于存储填充后的张量
    padded_tensors = []
    
    for tensor in tensor_group:
        x = tensor.shape[0]
        
        # 如果当前张量的 x 小于 x_max，进行填充
        if x < x_max:
            # 使用torch.nn.functional.pad在第0维进行填充
            padding = (0, 0, 0, 0, 0, 0, 0, x_max - x)  # 只在第0维填充 (x_max - x)
            padded_tensor = torch.nn.functional.pad(tensor, padding, mode='constant', value=0)
        else:
            padded_tensor = tensor  # 如果已经是 x_max 的大小，无需填充
        
        padded_tensors.append(padded_tensor)
    
    # 按第一个维度进行堆叠，形成 (8, x_max, 3, 1920, 1080) 的张量
    stacked_tensor = torch.stack(padded_tensors)
    
    return stacked_tensor

def pad_ground_tensors(tensor_group):
    # 找到这一组张量的 x_max
    x_max = max(tensor.shape[0] for tensor in tensor_group)
    
    # 创建一个列表，用于存储填充后的张量
    padded_tensors = []
    
    for tensor in tensor_group:
        x, m, n = tensor.shape
        
        # 如果当前张量的 x 小于 x_max，进行填充
        if x < x_max:
            # 使用torch.nn.functional.pad在第0维进行填充
            padded_tensor = torch.cat([tensor, torch.zeros(((x_max-x), m, n)).float()], dim=0)
        
            padded_tensors.append(padded_tensor)
        else:
            padded_tensors.append(tensor)
    
    # 按第一个维度进行堆叠，形成 (8, x_max, 3, 1920, 1080) 的张量
    stacked_tensor = torch.stack(padded_tensors)
    
    return stacked_tensor

def layer_per_pro(root, ids2nodes, nodes2ids, edges, root_word=""):
    if not root_word:
        root_word = f'{str(root["name"])}_{len(ids2nodes)}'
    if "children" in root:
        children = root["children"]
        for child in children:
            child_word = f'{str(child["name"])}_{len(ids2nodes)}'
            ids2nodes[len(ids2nodes)] = child_word
            nodes2ids[child_word] = len(nodes2ids)
            edges[f"{nodes2ids[root_word]}-{nodes2ids[child_word]}"] = child["role"]
            ids2nodes, nodes2ids, edges = layer_per_pro(child, ids2nodes, nodes2ids, edges, child_word)
    return ids2nodes, nodes2ids, edges

def tree2adj(tree_list):
    sense_list = []
    for pre_sense_tree in tree_list:
        root = pre_sense_tree["sememes"]
        real_root_word = f'{"|".join(str(root["name"]).split("|")[1:])}_0'
        sense_ids2nodes = {0:real_root_word}
        sense_nodes2ids = {real_root_word:0}
        sense_edges = {}
        ids2nodes, nodes2ids, edges = layer_per_pro(root, sense_ids2nodes, sense_nodes2ids, sense_edges, real_root_word)
        assert len(ids2nodes) == len(nodes2ids)
        init_adj = torch.zeros((len(ids2nodes), len(ids2nodes)))
        for edge in edges:
            i, j = edge.split("-")
            init_adj[int(i), int(j)] = 1
        nodes_list = [node for node in nodes2ids]
        assert len(nodes_list) == init_adj.shape[0]
        sense_list.append([nodes_list, init_adj.tolist()])
    # exit()
    return sense_list

def sememe_pro(w_token, w_list, tokenizer, hownet_dict=OpenHowNet.HowNetDict(), startToken="[CLS]"):
    # sentence = sent.replace("“", '"').replace("”", '"').replace("‘", "'").replace("…", "")
    # w_list = f"{startToken} {sentence}".split(" ")
    sent_adj_list = []
    sent_id_list = []
    # w_token = tokenizer.tokenize(" ".join(w_list))
    # w_list = w_list
    align = []
    curr_wrd = 0 # start at 1, b/c of CLS
    buf = []
    buf_o = []
    for i in range(len(w_token)): # ignore [SEP] final token
        if w_token[i].startswith("##"):
            strpd = w_token[i][2:]
            strpd_o = w_token[i] 
        else:
            strpd = w_token[i]
            strpd_o = w_token[i]
        buf.append(strpd)
        buf_o.append(strpd_o)
        fwrd = ''.join(buf)
        wrd = w_list[curr_wrd].lower()
        if fwrd == wrd or fwrd == "[UNK]":
            align.append([wrd, buf_o])
            curr_wrd += 1
            buf = []
            buf_o = []
    for item in align:
        word = item[0]
        word = word.lower()
        pretree_list = hownet_dict.get_sememes_by_word(word=word, display='dict')
        sememe_adj_list = tree2adj(pretree_list)
        senses_adj = []
        senses_id = []
        for sememe_adj in sememe_adj_list:
            sememe = sememe_adj[0]
            adj = sememe_adj[-1]
            sememes_id = []
            for item in sememe:
                item = str(item)
                if "|" in item:
                    sememes_id.append(tokenizer.convert_tokens_to_ids(tokenizer.tokenize(f"{startToken} {item.split('|')[1].split('_')[0]}")[1:]))
                else:
                    sememes_id.append(tokenizer.convert_tokens_to_ids(tokenizer.tokenize(f"{startToken} {item.split('_')[0]}")[1:]))
            if sememes_id:
                senses_adj.append(adj)
                senses_id.append(sememes_id)
            else:
                senses_adj.append([[]])
                senses_id.append([[]])
        if senses_id:
            sent_adj_list.append(senses_adj)
            sent_id_list.append(senses_id)
        else:
            sent_adj_list.append([[[]]])
            sent_id_list.append([[[]]])
    return sent_id_list, sent_adj_list

def get_layer_max_lengths(nested_list):
    """
    获取嵌套列表每一层的最大长度。
    :param nested_list: 多层嵌套列表。
    :return: 每层最大长度组成的列表。
    """
    if not isinstance(nested_list, list):
        return []
    
    # 当前层的最大长度
    max_length = len(nested_list)
    
    # 递归获取子层的最大长度
    sub_max_lengths = [
        get_layer_max_lengths(sublist) for sublist in nested_list if isinstance(sublist, list)
    ]
    
    # 如果子层存在，逐层取最大值
    if sub_max_lengths:
        max_lengths_per_layer = [max(layer) for layer in zip(*sub_max_lengths)]
        return [max_length] + max_lengths_per_layer
    else:
        return [max_length]
    
def recursive_padding(nested_list, padding_value=0):
    """
    对多层嵌套列表进行递归 padding。
    :param nested_list: 多层嵌套列表。
    :param padding_value: 填充值，默认为 0。
    :return: padding 后的嵌套列表。
    """
    def get_max_shape(nested_list):
        """获取嵌套列表的最大形状"""
        if not isinstance(nested_list, list):
            return []
        max_length = len(nested_list)
        max_subshape = [get_max_shape(sublist) for sublist in nested_list if isinstance(sublist, list)]
        return [max_length] + [max(max_dim) for max_dim in zip(*max_subshape)] if max_subshape else [max_length]
    
    def pad_to_length(lst, target_length, padding_value):
        """将当前列表填充到目标长度"""
        return lst + [padding_value] * (target_length - len(lst))
    
    def recursive_pad(nested_list, max_shape):
        """递归填充嵌套列表到最大形状"""
        if not isinstance(nested_list, list):
            return nested_list
        if len(max_shape) == 1:
            return pad_to_length(nested_list, max_shape[0], padding_value)
        return [
            recursive_pad(sublist, max_shape[1:]) if isinstance(sublist, list) else [padding_value] * max_shape[1]
            for sublist in pad_to_length(nested_list, max_shape[0], [])
        ]
    
    # 获取嵌套列表的最大形状
    max_shape = get_max_shape(nested_list)
    # 填充列表到最大形状
    return recursive_pad(nested_list, max_shape)


def collate_fn(data):
    bert_inputs, bert_inputs_token, bert_inputs_mask, images, pieces2word, labels, sent_length, sememes, adjs, sents = map(list, zip(*data))

    batch_size = len(bert_inputs)
    max_tok = np.max(sent_length)
    sent_length = torch.LongTensor(sent_length)
    labels = torch.LongTensor(labels)
    max_pie = np.max([x.shape[0] for x in bert_inputs_token])
    sememes = recursive_padding(sememes)
    sememes = torch.LongTensor(sememes) # bs * sent_l * sense_n * sememe_n * sememe_l
    adjs = recursive_padding(adjs)
    adjs = torch.LongTensor(adjs) # bs * sent_l * sense_n * sememe_n * sememe_n
    bert_inputs = pad_sequence(bert_inputs, True)
    bert_inputs_token = pad_sequence(bert_inputs_token, True)
    bert_inputs_mask = pad_sequence(bert_inputs_mask, True)
    images = pad_and_stack_tensors(images)

    def fill(data, new_data):
        for j, x in enumerate(data):
            new_data[j, :x.shape[0], :x.shape[1]] = x
        return new_data
    
    sub_mat = torch.zeros((batch_size, max_tok, max_pie), dtype=torch.bool)
    pieces2word = fill(pieces2word, sub_mat)

    return bert_inputs, bert_inputs_token, bert_inputs_mask, images, pieces2word, labels, sent_length, sememes, adjs, sents


class RelationDataset(Dataset):
    def __init__(self, bert_inputs, bert_inputs_token, bert_inputs_mask, images, pieces2word, labels, sent_length, sememes, adjs, sents):
        self.bert_inputs = bert_inputs
        self.bert_inputs_token = bert_inputs_token
        self.bert_inputs_mask = bert_inputs_mask
        self.images = images
        self.labels =labels
        self.pieces2word = pieces2word
        self.sent_length = sent_length
        self.sememes = sememes
        self.adjs = adjs
        self.sents = sents


    def __getitem__(self, item):
        return torch.LongTensor(self.bert_inputs[item]), \
               torch.LongTensor(self.bert_inputs_token[item]), \
               torch.LongTensor(self.bert_inputs_mask[item]), \
               torch.LongTensor(self.images[item]), \
               torch.LongTensor(self.pieces2word[item]), \
               self.labels[item], \
               self.sent_length[item], \
               self.sememes[item], \
               self.adjs[item], \
               self.sents[item]
               
               

    def __len__(self):
        return len(self.bert_inputs)



def process_bert(data, tokenizer, config):
    bert_inputs = []
    bert_inputs_token = []
    labels = []
    bert_inputs_mask = []
    images = []
    adjs = []
    sememes = []
    pieces2word = []
    sent_length = []
    max_len = 0
    sents = []
    for instance in tqdm(data[:10]):
        sent = f"{instance['text']}。{instance['text_discription']}。{instance['meme_discription']}"
        # sent = f"{instance['text']}"
        sent_list = re.findall(r'[\u4e00-\u9fa5]|[a-zA-Z]+|\d+|[^\w\s]', sent)
        if not config.is_b:
            label = instance["type"]
        else:
            label = instance["new_label"]
        tokens_token = [tokenizer.tokenize(word) for word in sent_list]
        pieces_token = [piece for pieces in tokens_token for piece in pieces]
        _bert_inputs_token = tokenizer.convert_tokens_to_ids(pieces_token)
        _bert_inputs_token = np.array([tokenizer.cls_token_id] + _bert_inputs_token + [tokenizer.sep_token_id])
        sent_id_list, sent_adj_list = sememe_pro(pieces_token, sent_list, tokenizer)
        length = len(sent_list)
        _pieces2word = np.zeros((length, len(_bert_inputs_token)), dtype=np.bool_)

        if tokenizer is not None:
            start = 0
            for i, pieces in enumerate(tokens_token):
                if len(pieces) == 0:
                    continue
                pieces = list(range(start, start + len(pieces)))
                _pieces2word[i, pieces[0] + 1:pieces[-1] + 2] = 1
                start += len(pieces)


        tokens = [["[CLS]"]] + [[word] for word in sent_list] + [["[SEP]"]]
        pieces = [piece for pieces in tokens for piece in pieces]

        _bert_inputs = tokenizer.convert_tokens_to_ids(pieces)

        _ent_output_mask, _bert_inputs_mask = [1] * len(_bert_inputs), [1] * len(_bert_inputs)


        video_id = instance['new_path']
        image = cv2.imread(f'./meme/{video_id}')
        image = cv2.resize(image, (config.height, config.width), interpolation=cv2.INTER_LANCZOS4)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        _image = torch.tensor(image).permute(2, 0, 1).float() / 255.0
        _image = _image.long()

        if _bert_inputs.__len__() > max_len:
            max_len = _bert_inputs.__len__()


        sent_length.append(length)      
        bert_inputs.append(_bert_inputs)
        bert_inputs_token.append(_bert_inputs_token)
        pieces2word.append(_pieces2word)
        bert_inputs_mask.append(_bert_inputs_mask)
        images.append(_image)
        labels.append(label)
        sememes.append(sent_id_list)
        adjs.append(sent_adj_list)
        sents.append(sent)

    return bert_inputs, bert_inputs_token, bert_inputs_mask, images, pieces2word, labels, sent_length,sememes, adjs, sents

def fill_vocab(dataset):
    Harmful, Non_harmful, Targeted_Harmful, Sexual_Innuendo, General_Offense, Dispirited_Culture = 0, 0, 0, 0, 0, 0
    for item in dataset:
        label = item["label"]
        t = item["type"]
        if t == 0:
            Harmful += 1
        elif t == 1:
            Non_harmful += 1
            Targeted_Harmful += 1
        elif t == 2:
            Non_harmful += 1
            Sexual_Innuendo += 1
        elif t == 3:
            Non_harmful += 1
            General_Offense += 1
        elif t == 4:
            Non_harmful += 1
            Dispirited_Culture += 1
    return Harmful, Non_harmful, Targeted_Harmful, Sexual_Innuendo, General_Offense, Dispirited_Culture


def load_data_bert(config):
    with open(f'./data/train.json', 'r', encoding='utf-8') as f:
        train_data = json.loads(f.read())
    with open(f'./data/test.json', 'r', encoding='utf-8') as f:
        test_data = json.loads(f.read())

    tokenizer = None
    while tokenizer is None:
        try:
            # tokenizer = AutoTokenizer.from_pretrained(config.bert_name, cache_dir="./cache/")
            tokenizer = AutoTokenizer.from_pretrained(
                config.bert_name,
                max_length=config.max_sequence_length,
                padding="max_length",
                truncation=True,
            )
        except (requests.exceptions.ConnectionError, requests.exceptions.Timeout):
            continue


    vocab = Vocabulary()
    train_Harmful, train_Non_harmful, train_Targeted_Harmful, train_Sexual_Innuendo, train_train_General_Offense, train_Dispirited_Culture = fill_vocab(train_data)
    test_Harmful, test_Non_harmful, test_Targeted_Harmful, test_Sexual_Innuendo, test_General_Offense, test_Dispirited_Culture = fill_vocab(test_data)

    table = pt.PrettyTable(["", "docs", "Harmful", "Non", "Targeted", "Sexual", "General", "Dispirited"])
    table.add_row(['train', len(train_data), train_Harmful, train_Non_harmful, train_Targeted_Harmful, train_Sexual_Innuendo, train_train_General_Offense, train_Dispirited_Culture])
    table.add_row(['test', len(test_data), test_Harmful, test_Non_harmful, test_Targeted_Harmful, test_Sexual_Innuendo, test_General_Offense, test_Dispirited_Culture])
    config.logger.info("\n{}".format(table))
    config.maxGroundNum = 1
    config.word_num = len(vocab.token2id)
    config.char_num = len(vocab.char2id)
    config.vocab = vocab
    train_dataset = RelationDataset(*process_bert(train_data, tokenizer, config))
    test_dataset = RelationDataset(*process_bert(test_data, tokenizer, config))
    return train_dataset, test_dataset


def load_embedding(config):
    vocab = config.vocab
    wvmodel = KeyedVectors.load_word2vec_format(config.embedding_path, binary=True)
    embed_size = config.word_emb_size
    embedding = np.random.uniform(-0.01, 0.01, (len(vocab), embed_size))
    hit = 0
    for token, i in vocab.token2id.items():
        if token in wvmodel:
            hit += 1
            embedding[i, :] = wvmodel[token]
    print("Total hit: {} rate {:.4f}".format(hit, hit / len(vocab)))
    embedding[0] = np.zeros(embed_size)
    embedding = torch.FloatTensor(embedding)
    return embedding
