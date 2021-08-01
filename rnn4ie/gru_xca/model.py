import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.nn import functional as F
from rnn4ie.util import crf

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class GRUXCAIE(nn.Module):
    def __init__(self, num_embeddings, embedding_dim,
                 hid_dim, num_layers, bidirectional,
                 num_class, dropout, n_heads, PAD_IDX, use_crf=True):
        super(GRUXCAIE, self).__init__()
        # Hyper-parameters
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.hid_dim = hid_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.num_class = num_class
        self.dropout = dropout
        self.n_heads = n_heads
        self.use_crf = use_crf

        # Components
        self.embedding = nn.Embedding(num_embeddings, embedding_dim, padding_idx=PAD_IDX)
        self.gru = nn.GRU(embedding_dim, hid_dim, num_layers,
                            batch_first=True,
                            bidirectional=bidirectional, dropout=dropout)
        fc_in_dim = hid_dim * 2 if bidirectional else hid_dim
        self.xca = XCA(fc_in_dim, n_heads, dropout)
        self.fc = nn.Linear(fc_in_dim, num_class)
        if use_crf:
            self.crf = crf.CRF(num_class, batch_first=True)

    def forward(self, padded_input, input_lengths):
        '''
        :param padded_input: [batch_size, seq_len]
        :param input_lengths: [batch_size]
        :return:
        '''
        padded_input = self.embedding(padded_input)  # [batch_size, seq_len, embedding_dim]
        total_length = padded_input.size(1)  # get the max sequence length
        packed_input = pack_padded_sequence(padded_input, input_lengths.cpu(),
                                            batch_first=True)
        packed_output, _ = self.gru(packed_input)
        output, _ = pad_packed_sequence(packed_output,
                                        batch_first=True,
                                        total_length=total_length)
        xca_output = self.xca(output) #[batch_size, seq_len, hid_dim]
        # Output Layer
        output = self.fc(xca_output)
        if self.use_crf:
            output = self.crf.decode(output)
        return output

    def log_likelihood(self, source, target, input_lengths):
        '''
        :param source: [batch_size, src_len]
        :param target: [batch_size, src_len]
        :param input_lengths: [batch_size]
        :return:
        '''
        padded_input = self.embedding(source)  # [batch_size, seq_len, embedding_dim]
        total_length = padded_input.size(1)  # get the max sequence length
        packed_input = pack_padded_sequence(padded_input, input_lengths.cpu(),
                                            batch_first=True)
        packed_output, _ = self.gru(packed_input)
        output, _ = pad_packed_sequence(packed_output,
                                        batch_first=True,
                                        total_length=total_length)
        xca_output = self.xca(output)  # [batch_size, seq_len, hid_dim]
        # Output Layer
        output = self.fc(xca_output)

        return -self.crf(output, target)


class XCA(nn.Module):
    def __init__(self, hid_dim, n_heads, dropout):
        super(XCA, self).__init__()
        assert hid_dim % n_heads == 0
        # hyper-parameter
        self.n_heads = n_heads
        self.temperature = nn.Parameter(torch.ones(n_heads, 1, 1))

        self.QKV = nn.Linear(hid_dim, hid_dim * 3)
        self.Z = nn.Linear(hid_dim, hid_dim)
        self.dropout = nn.Dropout(dropout)
        self.scale = torch.sqrt(torch.FloatTensor([hid_dim])).to(DEVICE)

    def forward(self, x):
        batch_size, seq_len, hid_dim = x.shape
        qkv = self.QKV(x).reshape(batch_size, seq_len, 3, self.n_heads, hid_dim // self.n_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        Q, K, V = qkv[0], qkv[1], qkv[2]
        # 转置
        Q = Q.transpose(-2, -1)
        K = K.transpose(-2, -1)
        V = V.transpose(-2, -1)

        Q = F.normalize(Q, dim=-1)
        K = F.normalize(K, dim=-1)

        weights = ((Q @ K.transpose(-2, -1)) / self.scale) * self.temperature
        attention = torch.softmax(weights, dim=-1)
        x = (self.dropout(attention) @ V)
        x = x.permute(0, 3, 1, 2).contiguous()
        x = x.view(batch_size, seq_len, hid_dim)
        x = self.Z(x)
        return x  # [batch_size, seq_len, hid_dim]