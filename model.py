from torch.nn import CrossEntropyLoss
import torch
import torch.nn as nn
from transformers import AutoModel, ViTModel
from utils import *
import copy
import torch.nn.functional as F
import math
import random

class MLP(nn.Module):
    def __init__(self, n_in, n_out, dropout=0):
        super().__init__()

        self.linear = nn.Linear(n_in, n_out)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.dropout(x)
        x = self.linear(x)
        x = self.activation(x)
        return x
    
class features_attention(nn.Module):
    def __init__(self, config, feature_attention=True):
        super(features_attention, self).__init__()
        self.config = config
        self.feature_attention = feature_attention
        if feature_attention:
            self.query = nn.Linear(config.latent, config.latent)
            self.key = nn.Linear(config.latent, config.latent)
            self.value = nn.Linear(config.latent, config.latent)
        else:
            self.query = nn.Linear(config.latent, config.latent)
            self.key = MLP(config.latent, config.latent, config.dropout)
            self.value = MLP(config.latent, config.latent, config.dropout)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, main_feature, feature, is_img=False):
        """
            main_feature: [batch_size, N1, hidden_dim]
            feature: [batch_size, N2, hidden_dim]
        """
        if self.feature_attention:
            query_layer = self.query(main_feature.float()) # [batch_size, N2, hidden_dim]
            key_layer = self.key(feature.float()) # [batch_size, N1, hidden_dim]
            value_layer = self.value(feature.float()) # [batch_size, N1, hidden_dim]
            attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))  # [bs, N2, N1]
            attention_probs = nn.Softmax(dim=-1)(attention_scores) # [bs, N2]
            attention_probs = self.dropout(attention_probs) # [bs, N2]
            attention_result = torch.matmul(attention_probs, value_layer) # [bs, N2, hidden_dim]
            return attention_result
        else:
            new_main_feature = torch.zeros_like(main_feature).to(self.config.device)
            new_main_feature[:, :, :] = main_feature[:, :, :]
            new_main_feature_head = torch.zeros_like(main_feature[:, 0, :].unsqueeze(1).float()).to(self.config.device)
            new_main_feature_head[:, 0, :] = main_feature[:, 0, :]
            query_layer = self.query(new_main_feature_head.float()) # [batch_size, 1, hidden_dim]
            key_layer = self.key(feature.float()) # [batch_size, n, hidden_dim]
            value_layer = self.value(feature.float()) # [batch_size, n, hidden_dim]
            attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))  # [bs, 1, n]
            attention_probs = nn.Softmax(dim=-1)(attention_scores) # [bs, 1]
            attention_probs = self.dropout(attention_probs) # [bs, 1]
            attention_result = torch.matmul(attention_probs, value_layer) # [bs, 1, hidden_dim]
            new_main_feature[:, 0, :] = attention_result.squeeze(1)
            return new_main_feature

class FeedForwardLayer(nn.Module):

    def __init__(self, hidden_size, dropout=0.5):
        super(FeedForwardLayer, self).__init__()
        self.hidden_size = hidden_size
        self.dropout_rate = dropout
        self.intermediate_layer = IntermediateLayer(self.hidden_size)
        self.intermediate_output_layer = IntermediateOutputLayer(self.hidden_size, self.dropout_rate)

    def forward(self, hidden_states):
        hidden_states1 = self.intermediate_layer(hidden_states)
        hidden_states = self.intermediate_output_layer(hidden_states1, hidden_states)
        return hidden_states

class IntermediateLayer(nn.Module):

    def __init__(self, hidden_size):
        super(IntermediateLayer, self).__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = self.hidden_size // 2
        self.dense = nn.Linear(self.hidden_size, self.intermediate_size)
        self.activate_fn = nn.GELU()

    def forward(self, hidden_states):
        hidden_states = self.activate_fn(self.dense(hidden_states))
        return hidden_states


class IntermediateOutputLayer(nn.Module):

    def __init__(self, hidden_size, dropout=0.5):
        super(IntermediateOutputLayer, self).__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = self.hidden_size // 2
        self.dropout_rate = dropout
        self.dense = nn.Linear(self.intermediate_size, self.hidden_size)
        self.layer_norm = nn.LayerNorm(self.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(self.dropout_rate)

    def forward(self, hidden_states, inputs):
        hidden_states = self.dropout(self.dense(hidden_states))
        hidden_states = self.layer_norm(hidden_states + inputs)
        return hidden_states

def clone_module(module: nn.Module, n):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(n)])

class CrossMultiHeadAttention(nn.Module):

    def __init__(self, hidden_size, num_heads=16, dropout_rate=0.5):
        super(CrossMultiHeadAttention, self).__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads

        assert self.hidden_size % self.num_heads == 0
        self.head_size = self.hidden_size // self.num_heads
        self.dropout_rate = dropout_rate
        self.dropout = nn.Dropout(self.dropout_rate)
        self.linears = clone_module(nn.Linear(self.hidden_size, self.hidden_size), 3)

    def forward(self, input_q, input_k, input_v):
        bs, sl1, hs = input_q.shape
        _, sl2, hs = input_k.shape
        assert hs == self.hidden_size

        q, k, v = [
            layer(x).view(bs, -1, self.num_heads, self.head_size).transpose(1, 2)
            for layer, x in zip(self.linears, (input_q, input_k, input_v))
        ]

        # score_masks = (1 - attn_masks1) * -1e30
        # score_masks = score_masks.unsqueeze(dim=-1).unsqueeze(dim=-1)
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_size)
        # attn_scores = attn_scores + score_masks
        attn_probs = nn.Softmax(dim=-1)(attn_scores)
        attn_probs = self.dropout(attn_probs)

        context_output = torch.matmul(attn_probs, v)
        context_output = context_output.permute(0, 2, 1, 3).contiguous()
        context_output = context_output.view(bs, -1, self.hidden_size)

        return context_output, attn_probs
    
class TextImageFusionModule(nn.Module):
    def __init__(self, hidden_dim, num_heads=16):
        super(TextImageFusionModule, self).__init__()

        self.linear_text_to_image = nn.Linear(hidden_dim, hidden_dim * 2)  # 文本嵌入到图像空间的线性映射

        self.attention = CrossMultiHeadAttention(hidden_size=hidden_dim, num_heads=num_heads)

        self.feedforward = FeedForwardLayer(hidden_dim)

        self.ln1 = nn.LayerNorm(hidden_dim)
        self.ln2 = nn.LayerNorm(hidden_dim)

    def forward(self, fusion_embeddings, embeddings):
        batch_size, image_num, hidden_dim = embeddings.size()

        fusion_embeddings_k, fusion_embeddings_v = torch.split(self.linear_text_to_image(fusion_embeddings), hidden_dim, dim=-1)  # [batch, length, hidden_dim]
        # 将文本嵌入扩展为 K 和 V
        K = torch.cat([fusion_embeddings_k, embeddings], dim=1)  # [batch, length+imgNum, hidden_dim]
        V = torch.cat([fusion_embeddings_v, embeddings], dim=1)

        Q = embeddings
        # 多头注意力
        attn_output, _ = self.attention(Q, K, V)  # 输出: [batch * image_num, patch_num, hidden_dim]

        # 残差连接和层归一化
        attn_output = self.ln1(attn_output + Q)

        # 前馈神经网络
        ff_output = self.feedforward(attn_output)

        # 残差连接和层归一化
        output = self.ln2(ff_output + attn_output)
        return output

class GCN(nn.Module):
    def  __init__(self, nfeat, nhid, nclass=0, dropout=0.1):
        super(GCN, self).__init__()

        self.gc1 = GraphConvolution(nfeat, nhid)
        # self.gc2 = GraphConvolution(nhid, nclass)
        self.dropout = dropout

    def forward(self, x, adj):    #x特征矩阵,agj邻接矩阵 
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        # x = self.gc2(x, adj)
        # return F.log_softmax(x, dim=1)
        return x

class GraphConvolution(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.einsum('blh,hh->blh', input, self.weight)
        output = torch.einsum('bll,blh->blh', adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

class SememeEmbedding(nn.Module):
    def __init__(self, config, hidden_size, use_sememe):
        super(SememeEmbedding, self).__init__()
        self.use_sememe = use_sememe
        self.hidden_size = hidden_size
        self.gcn = GCN(hidden_size, hidden_size)
        self.config = config

    def forward(self, sent_embedding, sememes, adjs):
        pass


           
# class SememeEmbedding(nn.Module):
#     def __init__(self, config, hidden_size, use_sememe, roberta_embedding):
#         super(SememeEmbedding, self).__init__()
#         self.embedding = roberta_embedding
#         self.use_sememe = use_sememe
#         self.hidden_size = hidden_size
#         self.gcn = GCN(hidden_size, hidden_size)
#         self.config = config

#     def forward(self, sent_embedding, sent_l, sememes, adjs):
#         sememe_fusion_feature = torch.zeros_like(sent_embedding).to(self.config.device)
#         for bs in range(sent_embedding.shape[0]):
#             sememe = sememes[bs]
#             length = sent_l[bs]
#             assert len(sememe) == length
#             fusion_word = []
#             for word_id, words_sememe in enumerate(sememe):
#                 s = sent_embedding[bs, word_id, :].unsqueeze(0)
#                 senses_id_list = words_sememe[1]
#                 if self.use_sememe:
#                     if len(senses_id_list) != 0:
#                         sememe_tensor = []
#                         for sense in senses_id_list:
#                             nodes = sense[0]
#                             adj = torch.FloatTensor(sense[1]).to(self.config.device)
#                             nodes_tensor = []
#                             for node_tokens in nodes:
#                                 if not node_tokens:
#                                     node_tokens.append(0)
#                                 node_tensor = torch.LongTensor(node_tokens).unsqueeze(0).to(self.config.device) # 1*len
#                                 node_embedding = self.embedding(node_tensor)[0] #1*len*768
#                                 node_embedding = torch.mean(node_embedding, dim=1) #1*768
#                                 nodes_tensor.append(node_embedding)
#                             nodes_tensor = torch.cat(nodes_tensor, dim=0)
#                             assert adj.shape[0] == nodes_tensor.shape[0]
#                             out = self.gcn(nodes_tensor, adj)[0, :]
#                             sememe_tensor.append(out)
#                         sememe_tensor = torch.stack(sememe_tensor, dim=0)
#                         distance = F.pairwise_distance(s, sememe_tensor, p=2)
#                         attentionSocre = torch.softmax(distance, dim=0)
#                         attentionSememeTensor = torch.einsum("a,ab->ab", attentionSocre, sememe_tensor)
#                         fusion_word.append(torch.cat([attentionSememeTensor.mean(0).unsqueeze(0), s], dim=0).mean(0))
                        
#                 else:
#                     fusion_word.append(s.mean(dim=0))
#             assert len(fusion_word) == length
#             sememe_fusion_feature[bs, :length, :] = torch.stack(fusion_word, dim=0)
#         return sememe_fusion_feature

class Model(nn.Module):
    def __init__(self, config):
        super(Model, self).__init__()

        self.config = config
        self.height=config.height
        self.width=config.width
        self.bert_hid_size=config.bert_hid_size
        self.label_num = len(config.vocab.label2id[config.is_b])

        self.bert = AutoModel.from_pretrained(
                    config.bert_name, 
                    max_length=config.max_sequence_length,
                    output_hidden_states=True
                )

        self.vit = ViTModel.from_pretrained(config.vit_name, output_hidden_states=True, use_mask_token=True)

        self.FFN = FeedForwardLayer(config.bert_hid_size)

        if config.is_h:
            self.attention = nn.ModuleDict({
                f"{i}": TextImageFusionModule(config.bert_hid_size) for i in range(config.num_hidden_layers)})
        else:
            config.is_decoder = False
            self.attention = BertAttention(config)
        
        self.text_fusion_weight = nn.Parameter(torch.randn(config.num_hidden_layers+1, 1))

        self.output = nn.Linear(config.bert_hid_size, self.label_num)

        torch.nn.init.orthogonal_(self.output.weight, gain=1)
        self.gcn = GCN(config.bert_hid_size, config.bert_hid_size)

    def piece2word(self, piece_embs, pieces2word):
        length = pieces2word.size(1)

        min_value = torch.min(piece_embs).item()

        # Max pooling word representations from pieces
        _piece_embs = piece_embs.unsqueeze(1).expand(-1, length, -1, -1)
        _piece_embs = torch.masked_fill(_piece_embs, pieces2word.eq(0).unsqueeze(-1), min_value)
        word_embs, _ = torch.max(_piece_embs, dim=2)
        return word_embs
    
    def get_fusion_feature(self, hidden_outputs, embs_last):
        # hidden_outputs: bs * length * hidden or bs * imgNum * patchNum * hidden
        features = [embs_last]
        for i in range(1, len(hidden_outputs)):
            outputs = hidden_outputs[i] # bs*batchNum*hidden
            inputs_attention_embs = self.attention[f"{i-1}"](outputs, embs_last) #bs*len*768
            packed_outs = self.FFN(inputs_attention_embs) # bs*len*lstm_hidden   
            features.append(packed_outs)
        features = torch.einsum("asld,ab->bsld", torch.stack(features, dim=0), self.text_fusion_weight).squeeze(0) 
        return features
    
    def expand_and_fill_zero_dims(self, tensor):
        shape = list(tensor.size())
        # 将所有为 0 的维度替换为 1
        new_shape = [dim if dim != 0 else 1 for dim in shape]
        # 重新创建张量，并初始化为 0
        result = torch.zeros(new_shape, dtype=tensor.dtype, device=tensor.device)
        # 复制原张量内容到扩展后的张量
        slices = tuple(slice(0, dim) for dim in shape)  # 生成原张量对应的切片
        result[slices] = tensor
        return result

    def forward(self, bert_input, images, pieces2word, labels, sent_l, sememes, adjs, sents, is_train=True):
        """
            word embdding
        """

        inputs_embs = self.bert(bert_input, attention_mask=bert_input.ne(0).float())[0] # bs*len*768

        fusion = self.piece2word(inputs_embs, pieces2word)

        if self.config.use_sememe:
            if 0 in list(sememes.size()):
                sememes = self.expand_and_fill_zero_dims(sememes)
                adjs = self.expand_and_fill_zero_dims(adjs)
            
            _, sent_len, sense_num, sememe_num, sememe_len = sememes.shape

            sememes = sememes.view(-1, sememe_len)
            adjs = adjs.view(-1, sememe_num, sememe_num)

            sememes_embs = self.bert.embeddings.word_embeddings(sememes) #[bs*sent_len*sense_num*sememe_num,sememe_len,768]
            sememes_embs = self.gcn(sememes_embs.mean(1).view(-1, sememe_num, self.bert_hid_size), adjs)[:, 0, :].view(-1, sent_len, sense_num, self.bert_hid_size)
            distance = torch.einsum("bld,blnd->bln", fusion, sememes_embs)
            attentionSocre = torch.softmax(distance, dim=0)
            attentionSememeTensor = torch.einsum("bln,blnd->blnd", attentionSocre, sememes_embs)
            fusion = fusion + attentionSememeTensor.mean(2)

        if self.config.is_h:
            img_embs_hidden_states = self.vit(images).hidden_states # bs , patchNum , hidden
            fusion = self.get_fusion_feature(img_embs_hidden_states, fusion)
        
        logits = self.output(torch.mean(fusion, dim=1))

        """
            entity preds
        """ 
        if is_train:
            if self.config.is_b:
                loss_fct = nn.BCEWithLogitsLoss()
                labels = labels.float()
            else:
                loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits, labels) 
            return loss
        else:
            result = {}
            outputs = torch.argmax(logits, dim=1).tolist()
            if self.config.is_b:
                labels = torch.argmax(labels, dim=1).tolist()
            else:
                labels = labels.tolist() 
            for i in range(self.label_num):
                result[i] = [outputs.count(i), labels.count(i), sum(1 for x, y in zip(outputs, labels) if x == i and y == i)]
            # result[-1] = [np.sum(np.numpy(outputs) > 0), np.sum(np.numpy(labels) > 0), sum(1 for x, y in zip(outputs, labels) if x > 0 and y > 0)]
            return result