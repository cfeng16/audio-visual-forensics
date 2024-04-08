import torch.nn as nn
import torch 
import math
import copy
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict

class FeedForwardBlock(nn.Module):
    def __init__(self, inp_dim: int, hidden_dim_feedforward: int):
        super().__init__()

        """
        An implementation of the FeedForward block in the Transformers. We pass  
        the input through stacked 2 MLPs and 1 ReLU layer. The forward pass has  
        following architecture:
        
        linear - relu -linear
        
        The input will have a shape of (N, K, M) where N is the batch size, K is 
        the sequence length and M is the embedding dimension. 
        
        args:
            inp_dim: int representing embedding dimension of the input tensor
                     
            hidden_dim_feedforward: int representing the hidden dimension for
                the feedforward block
        """

        ##########################################################################
        # TODO: initialize two MLPs here with the first one using inp_dim as input
        # dimension and hidden_dim_feedforward as output and the second with     #
        # hidden_dim_feedforward as input. You should figure out the output      #
        # dimesion of the second MLP. Initialize the weights of all the MLPs     #
        # according to the strategy mentioned in SelfAttention block             #
        # HINT: Will the shape of input and output shape of the FeedForwardBlock #
        # change?                                                                #
        ##########################################################################
        # Replace "pass" statement with your code
        self.linear1 = nn.Linear(inp_dim, hidden_dim_feedforward)
        #torch.nn.init.uniform_(self.linear1.weight, -torch.sqrt(torch.tensor(6/(inp_dim+hidden_dim_feedforward))), torch.sqrt(torch.tensor(6/(inp_dim+hidden_dim_feedforward))))
        torch.nn.init.xavier_uniform_(self.linear1.weight, gain=1)
        self.non_linear = nn.ReLU()
        self.linear2 = nn.Linear(hidden_dim_feedforward, inp_dim)
        torch.nn.init.xavier_uniform_(self.linear2.weight, gain=1)
        #torch.nn.init.uniform_(self.linear2.weight, -torch.sqrt(torch.tensor(6/(inp_dim+hidden_dim_feedforward))), torch.sqrt(torch.tensor(6/(inp_dim+hidden_dim_feedforward))))
        ##########################################################################
        #               END OF YOUR CODE                                         #
        ##########################################################################

    def forward(self, x):
        """
        An implementation of the forward pass of the FeedForward block.

        args:
            x: a Tensor of shape (N, K, M) which is the output of
               MultiHeadAttention
        returns:
            y: a Tensor of shape (N, K, M)
        """
        y = None
        ###########################################################################
        # TODO: Use the two MLP layers initialized in the init function to perform#
        # a forward pass. You should be using a ReLU layer after the first MLP and#
        # no activation after the second MLP                                      #
        ###########################################################################
        # Replace "pass" statement with your code
        y = self.linear2(self.non_linear(self.linear1(x)))
        ##########################################################################
        #               END OF YOUR CODE                                         #
        ##########################################################################
        return y

def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class PositionwiseFeedForward(nn.Module):
    "Implements FFN equation."

    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(self.w_1(x).relu()))

class PositionalEncoding(nn.Module):
    "Implement the PE function."

    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[:, : x.size(1)].requires_grad_(False)
        return self.dropout(x)

class LayerNorm(nn.Module):
    "Construct a layernorm module (See citation for details)."

    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2

class SublayerConnection(nn.Module):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """

    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        "Apply residual connection to any sublayer with the same size."
        return x + self.dropout(sublayer(self.norm(x)))

class ConvCompress(nn.Module):
    def __init__(self, dim, ratio = 3, groups = 1):
        super().__init__()
        self.conv = nn.Conv1d(dim, dim, ratio, stride = ratio, groups = groups)

    def forward(self, mem):
        mem = mem.transpose(1, 2)
        compressed_mem = self.conv(mem)
        return compressed_mem.transpose(1, 2)

# main class

class PositionwiseFeedForward(nn.Module):
    "Implements FFN equation."

    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(self.w_1(x).relu()))


class MemoryCompressedAttention(nn.Module):
    def __init__(
        self,
        dim,
        heads = 8,
        causal = False,
        compression_factor = 3,
        dropout = 0.):
        super().__init__()
        assert (dim % heads) == 0, 'dimension must be divisible by number of heads'

        self.heads = heads
        self.causal = causal

        self.compression_factor = compression_factor
        self.compress_fn = ConvCompress(dim, compression_factor, groups = heads)

        self.to_qkv = nn.Linear(dim, dim * 3, bias = False)
        self.to_out = nn.Linear(dim, dim)

        self.dropout = nn.Dropout(dropout)

        self.null_k = nn.Parameter(torch.zeros(1, 1, dim))
        self.null_v = nn.Parameter(torch.zeros(1, 1, dim))

    def forward(self, x, input_mask = None):
        b, t, d, h, cf, device = *x.shape, self.heads, self.compression_factor, x.device
        q, k, v = self.to_qkv(x).chunk(3, dim=-1)

        # make sure keys and values sequence lengths
        # are divisible by the compression factor
        padding = cf - (t % cf)
        if padding < cf:
            k, v = map(lambda t: F.pad(t, (0, 0, padding, 0)), (k, v))

        # compress keys and values
        if self.compression_factor == 1:
            pass
        else:
            k, v = map(self.compress_fn, (k, v))
        batch_size = q.shape[0]
        # attach a null key and value, in the case that the first query has no keys to pay attention to
        null_k = self.null_k.repeat(batch_size, 1, 1)
        null_v = self.null_v.repeat(batch_size, 1, 1)
        k = torch.cat((null_k, k), dim=1)
        v = torch.cat((null_v, v), dim=1)

        # merge heads
        q, k, v = map(lambda t: t.reshape(*t.shape[:2], h, -1).transpose(1, 2), (q, k, v))

        # attention
        dots = torch.einsum('bhid,bhjd->bhij', q, k) * d ** -0.5
        #attn = dots.softmax(dim=-1)

        # causal masking, if needed
        if self.causal:
            mask_q = mask_k = torch.arange(t, device=device)

            if padding < cf:
                mask_k = F.pad(mask_k, (padding, 0))

            mask_k, _ = mask_k.reshape(-1, cf).max(dim=-1)
            mask = mask_q[:, None] < mask_k[None, :]
            mask = F.pad(mask, (1, 0), value=False)

            dots.masked_fill_(mask[None, None, ...], float('-inf'))
            del mask

        # input masking
        if input_mask is not None:
            mask_q = mask_k = input_mask
            if padding < cf:
                mask_k = F.pad(mask_k, (padding, 0), value=False)
            mask_k = mask_k.reshape(b, -1, cf).sum(dim=-1) > 0
            #mask = mask_q[:, None, :, None] < mask_k[:, None, None, :]
            mask_k = mask_k[:, None, None, :]
            mask = mask_k.repeat(1, 1,  mask_q.shape[1], 1)
            mask = F.pad(mask, (1, 0), value=False)

            dots.masked_fill_(mask, float('-inf'))
            del mask

        # dropout
        attn = dots.softmax(dim=-1)
        attn = self.dropout(attn)
        out = torch.einsum('bhij,bhjd->bhid', attn, v)

        # split heads and combine
        out = out.transpose(1, 2).reshape(b, t, d)
        return self.to_out(out)

class decoder_only_module(nn.Module):
    def __init__(self, input_dim, compress_factor, num_heads, dropout_prob):
        super().__init__()
        self.docoder = MemoryCompressedAttention(
        dim = input_dim,
        heads = num_heads,                 # number of heads
        causal = True,            # auto-regressive or not
        compression_factor = compress_factor,    # compression ratio
        dropout = dropout_prob,              # dropout post-attention
        )
        self.drop = nn.Dropout(dropout_prob)
        self.norm = LayerNorm(input_dim)
        self.ff = PositionwiseFeedForward(d_model=input_dim, d_ff= input_dim)

    def forward(self, x, mask=None):
        if mask is not None:
            out1 = self.docoder(x,input_mask=mask)
        else:
            out1 = self.docoder(x)
        out2 = self.drop(self.norm(out1 + x))
        out3 = self.ff(out2)
        x = self.drop(self.norm(out2+out3))

        '''
        if mask is not None:
            out4 = self.docoder(x, input_mask = mask)
        else:
            out4 = self.docoder(x)
        out5 = self.drop(self.norm(out4 + x))
        out6 = self.ff(out5)
        x = self.drop(self.norm(out5 + out6))
         
        if mask is not None:
            out7 = self.docoder(x, input_mask = mask)
        else:
            out7 = self.docoder(x)
        out8 = self.drop(self.norm(out7 + x))
        out9 = self.ff(out8)
        x = self.drop(self.norm(out8 + out9))
        
        '''
        return x

class transformer_decoder(nn.Module):
    def __init__(self, input_dim_old, input_dim, compress_factor, num_heads, dropout_prob, max_len, layers):
        super().__init__()
        self.decoders = nn.ModuleList()
        #self.decoder = OrderedDict()
        self.layers = layers
        #self.emb_layer = nn.Embedding(num_embeddings=input_dim_old, embedding_dim=input_dim)
        #for i in range(layers):
        self.decoder1 = decoder_only_module(input_dim, compress_factor, num_heads, dropout_prob)
        self.decoder2 = decoder_only_module(input_dim, compress_factor, num_heads, dropout_prob)
            #name = 'decoder' + '_' + str(i)
            #self.decoder[name]  =  decoder_only_module(input_dim, compress_factor, num_heads, dropout_prob)
            #self.decoders.append(decoder_only_module(input_dim, compress_factor, num_heads, dropout_prob))
        #self.decoder = nn.Sequential(self.decoder)
        #self.decoder3 = decoder_only_module(input_dim, compress_factor, num_heads, dropout_prob)
        self.lin1 = nn.Linear(input_dim_old, input_dim)
        self.lin2 = nn.Linear(input_dim, input_dim_old)
        torch.nn.init.normal_(self.lin1.weight, mean=0, std=math.sqrt(2 / input_dim_old))
        torch.nn.init.zeros_(self.lin1.bias)
        a = (6 / (input_dim + input_dim_old)) ** 0.5
        nn.init.uniform_(self.lin2.weight, -a, a)
        self.relu = nn.ReLU()
        #self.pos = PositionalEncoding(d_model=input_dim, dropout=dropout_prob, max_len=50)
        self.pos_embedding = nn.Parameter(torch.randn(1, max_len, input_dim))
    def forward(self, x, mask):
        
        x = self.lin1(x)
        x = self.relu(x)
        
        #x = self.emb_layer(x)
        #x = self.pos(x)
        x += self.pos_embedding
        x = self.decoder1(x, mask)
        x = self.decoder2(x, mask)
        # for j in range(self.layers):
        #     x = self.decoders[j](x, mask)
        #x = self.decoder3(x, mask)
        out = self.lin2(x)
        return out


#model = decoder_only(input_dim_old=31, input_dim=64)
#input = torch.randn(1, 20, 31)
#output = model(input)
#print(output)
