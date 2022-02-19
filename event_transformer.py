import torch
from torch import nn

import math
import ml_collections
from torch.nn.modules.conv import Conv2d
from torch.nn.modules.linear import Linear
from torch.nn.modules.loss import CrossEntropyLoss
from torch.nn.modules.normalization import LayerNorm
import copy

from torch.nn.modules.sparse import Embedding
from torch.nn.parameter import Parameter
# position embeding
def get_config():
    config = ml_collections.ConfigDict()
    config.patches = ml_collections.ConfigDict({'size':16})
    config.hidden_size = 256
    config.transformer = ml_collections.ConfigDict()
    config.transformer.mlp_dim = 2048
    config.transformer.num_heads = 4
    config.transformer.num_layers = 4
    config.transformer.attention_dropout_rate = 0
    config.transformer.dropout_rate = 0.1
    return config

class Attention(nn.Module):
    def __init__(self,config,vis):
        super(Attention,self).__init__()
        self.vis = vis
        self.num_attention_heads = config.transformer["num_heads"]
        self.attention_head_size = int(config.hidden_size / self.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size,self.all_head_size)
        self.key = nn.Linear(config.hidden_size,self.all_head_size)
        self.value = nn.Linear(config.hidden_size,self.all_head_size)
        self.out = nn.Linear(config.hidden_size,config.hidden_size)

        self.attn_dropout = nn.Dropout(config.transformer["attention_dropout_rate"])
        self.proj_dropout = nn.Dropout(config.transformer["attention_dropout_rate"])
        self.softmax = nn.Softmax(dim=-1)

    def transpose_for_scores(self,x):
        new_x_shape = x.size()[:-1]+(self.num_attention_heads,self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0,2,1,3)
    def forward(self,hidden_states):
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        query_layer  = self.transpose_for_scores(mixed_query_layer)
        key_layer  = self.transpose_for_scores(mixed_key_layer)
        value_layer  = self.transpose_for_scores(mixed_value_layer)

        attention_scores = torch.matmul(query_layer,key_layer.transpose(-1,-2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        attention_probs = self.softmax(attention_scores)
        weights = attention_probs if self.vis else None
        attention_probs = self.attn_dropout(attention_probs)

        context_layer = torch.matmul(attention_probs,value_layer)
        context_layer = context_layer.permute(0,2,1,3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        attention_output = self.out(context_layer)
        attention_output = self.proj_dropout(attention_output)

        return attention_output,weights
# class EventPrepare(nn.Module):
#     def __init__(self,config):
#         super(EventPrepare,self).__init__()
#         self.layer1 = nn.Sequential(nn.Conv2d(in_channels=1,out_channels=3,kernel_size=3),
#                                     nn.ReLU(),
#                                     nn.MaxPool2d(2),
#                                     nn.Conv2d(in_channels=3,out_channels=9,kernel_size=3),
#                                     nn.ReLU(),
#                                     nn.MaxPool2d(2))
#         self.layer2 = nn.Sequential(nn.Conv2d(),
#                                     nn.ReLU(),
#                                     nn.ConvTranspose2d(),
#                                     nn.Conv2d(),
#                                     nn.ReLU(),
#                                     nn.ConvTranspose2d())
#     def forward(self,x):
#         x = self.layer1(x)
#         x = self.layer2(x)
#         return x
class MLP(nn.Module):
    def __init__(self,config):
        super(MLP,self).__init__()
        self.fc1 = nn.Linear(config.hidden_size,config.transformer["mlp_dim"])
        self.fc2 = nn.Linear(config.transformer["mlp_dim"],config.hidden_size)
        self.act_fn = nn.GELU()
        self.dropout = nn.Dropout(config.transformer["dropout_rate"])

        self._init_weights()
    def _init_weights(self):
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.normal_(self.fc1.bias,std=1e-6)
        nn.init.normal_(self.fc2.bias,std=1e-6)
    
    def forward(self,x):
        x = self.fc1(x)
        x = self.act_fn(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x

class Block(nn.Module):
    def __init__(self,config,vis):
        super(Block,self).__init__()
        self.hidden_size = config.hidden_size
        self.attention_norm = LayerNorm(config.hidden_size,eps=1e-6)
        self.ffn_norm = LayerNorm(config.hidden_size,eps=1e-6)
        self.ffn = MLP(config)
        self.attn = Attention(config,vis)
    def forward(self,x):
        h = x
        x = self.attention_norm(x)
        x,weights = self.attn(x)
        x = x + h

        h = x
        x = self.ffn_norm(x)
        x = self.ffn(x)
        x = x + h
        return x,weights
class Encoder(nn.Module):
    def __init__(self,config,vis):
        super(Encoder,self).__init__()
        self.vis = vis
        self.layer = nn.ModuleList()
        self.encoder_norm = LayerNorm(config.hidden_size,eps=1e-6)
        for _ in range(config.transformer["num_layers"]):
            layer = Block(config,vis)
            self.layer.append(copy.deepcopy(layer))

    def forward(self,hidden_states):
        attn_weights = []
        for layer_block in self.layer:
            hidden_states,weights = layer_block(hidden_states)
            if self.vis:
                attn_weights.append(weights)
        encoded = self.encoder_norm(hidden_states)
        return encoded, attn_weights
class Transformer(nn.Module):
    def __init__(self,config,img_size,vis):
        super(Transformer,self).__init__()
        self.embeddings = Conv2d(in_channels=2,out_channels=config.hidden_size,kernel_size=36,stride=36)
        self.encoder = Encoder(config,vis)
        self.cls_token = Parameter(torch.zeros(1,1,config.hidden_size))

    def forward(self,input_ids):
        cls_tokens = self.cls_token.expand(input_ids.shape[0],-1,-1)
        x = self.embeddings(input_ids)
        x = x.flatten(2)
        embedding_output = x.transpose(-1,-2)
        embedding_output = torch.cat((cls_tokens,embedding_output),dim=1)
        # transform evnets to frame-like data trough pre-transform model
        encoded,attn_weights = self.encoder(embedding_output)
        return encoded,attn_weights
#  A eVENts and frame time aliGning encodER
class Avenger(nn.Module):
    def __init__(self,config,img_size=800,vis=False):
        super(Avenger,self).__init__()
        self.transformer = Transformer(config,img_size,vis)
        self.head = Linear(config.hidden_size,1)
        self.softmax = nn.Softmax(dim=0)

    def forward(self,x,labels=None):
        x,attn_weights = self.transformer(x)
        logits = self.head(x[:,0])[1:]
        # logits = self.softmax(logits)

        if labels is not None:
            labels = labels.unsqueeze(0).softmax(dim=1)
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.transpose(-1,-2),labels)
            return loss
        else:
            return logits,attn_weights
if __name__ == "__main__":
    config = get_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(44)
    img = torch.rand(24,2,800,800)
    # attention = Attention(config,vis=True)
    # out_self_attention,_ = attention(img)
    # print(out_self_attention.shape)
    # mlp = MLP(config)
    # out_mlp = mlp(out_self_attention)
    # print(out_mlp.shape)
    # block = Block(config,vis=True)
    # out_block,_=block(img)
    # print(out_block.shape)
    # encoder = Encoder(config,vis=True)
    # out_encoder,_ = encoder(img)
    # print(out_encoder[:,0].shape)
    aven = Avenger(config,vis=True)
    aven.to(device)
    img = img.to(device)
    output,_ = aven(img)
    print(output.shape)