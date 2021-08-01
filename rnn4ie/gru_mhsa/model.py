import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from rnn4ie.util import crf

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class GRUMHSAIE(nn.Module):
    def __init__(self, num_embeddings, embedding_dim,
                 hid_dim, num_layers, bidirectional,
                 num_class, dropout, n_heads, PAD_IDX, use_crf=True):
        super(GRUMHSAIE, self).__init__()
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
        self.bigru = nn.GRU(embedding_dim, hid_dim, num_layers,
                            batch_first=True,
                            bidirectional=bidirectional, dropout=dropout)
        fc_in_dim = hid_dim * 2 if bidirectional else hid_dim
        self.unigru = nn.GRU(fc_in_dim, fc_in_dim, num_layers=1, batch_first=True, bidirectional=False, dropout=dropout)
        self.attention = Attention(fc_in_dim, n_heads, dropout)
        self.fc_attn = nn.Linear(fc_in_dim, fc_in_dim)
        self.fc_attn_fc = nn.Linear(fc_in_dim, fc_in_dim)
        self.fc_ht = nn.Linear(fc_in_dim, fc_in_dim)
        self.fc_out = nn.Linear(fc_in_dim, num_class)
        if use_crf:
            self.crf = crf.CRF(num_class, batch_first=True)

    def make_mask(self, input):
        input_mask = (input != 0).unsqueeze(1).unsqueeze(2)
        return input_mask

    def forward(self, padded_input, input_lengths):
        '''
        :param padded_input: [batch_size, seq_len]
        :param input_lengths: [batch_size]
        :return:
        '''
        mask_input = self.make_mask(padded_input)  # [batch_size, 1, 1, seq_leg]
        padded_input = self.embedding(padded_input)  # [batch_size, seq_len, embedding_dim]
        # GRU Layers
        total_length = padded_input.size(1)  # get the max sequence length
        packed_input = pack_padded_sequence(padded_input, input_lengths.cpu(), batch_first=True)
        packed_output, _ = self.bigru(packed_input)
        bigru_output, _ = pad_packed_sequence(packed_output,
                                              batch_first=True,
                                              total_length=total_length)  # [batch_size, seq_len, hid_dim*2]
        unigru_output, _ = self.unigru(bigru_output)  # [batch_size, seq_len, hid_dim*2]
        # attention
        context = self.attention(unigru_output, bigru_output, bigru_output, mask_input)
        context_fc = self.fc_attn(context)
        context_fc_fc = self.fc_attn_fc(context_fc)
        ht_fc = self.fc_ht(bigru_output)
        ft = context_fc * torch.sigmoid(context_fc_fc + ht_fc) + bigru_output
        output = self.fc_out(ft)
        if self.use_crf:
            output = self.crf.decode(output)
        return output  # [batch_size, seq_len, num_class]

    def log_likelihood(self, source, target, input_lengths):
        '''
        :param source: [batch_size, src_len]
        :param target: [batch_size, src_len]
        :param input_lengths: [batch_size]
        :return:
        '''
        mask_input = self.make_mask(source)  # [batch_size, 1, 1, seq_leg]
        padded_input = self.embedding(source)  # [batch_size, seq_len, embedding_dim]
        # GRU Layers
        total_length = padded_input.size(1)  # get the max sequence length
        packed_input = pack_padded_sequence(padded_input, input_lengths.cpu(), batch_first=True)
        packed_output, _ = self.bigru(packed_input)
        bigru_output, _ = pad_packed_sequence(packed_output,
                                              batch_first=True,
                                              total_length=total_length)  # [batch_size, seq_len, hid_dim*2]
        unigru_output, _ = self.unigru(bigru_output)  # [batch_size, seq_len, hid_dim*2]
        # attention
        context = self.attention(unigru_output, bigru_output, bigru_output, mask_input)
        context_fc = self.fc_attn(context)
        context_fc_fc = self.fc_attn_fc(context_fc)
        ht_fc = self.fc_ht(bigru_output)
        ft = context_fc * torch.sigmoid(context_fc_fc + ht_fc) + bigru_output
        output = self.fc_out(ft)

        return -self.crf(output, target)

class Attention(nn.Module):
    def __init__(self, hid_dim, n_heads, dropout):
        super(Attention, self).__init__()
        assert hid_dim % n_heads == 0
        # hyper-parameter
        self.hid_dim = hid_dim
        self.n_heads = n_heads
        self.head_dim = hid_dim // n_heads
        # components
        self.Q = nn.Linear(hid_dim, hid_dim)
        self.K = nn.Linear(hid_dim, hid_dim)
        self.V = nn.Linear(hid_dim, hid_dim)
        self.Z = nn.Linear(hid_dim, hid_dim)
        self.dropout = nn.Dropout(dropout)
        self.scale = torch.sqrt(torch.FloatTensor([self.hid_dim])).to(DEVICE)

    def forward(self, q, k, v, mask=None):
        batch_size = q.shape[0]
        Q = self.Q(q)
        K = self.K(k)
        V = self.V(v)

        Q = Q.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        K = K.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        V = V.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)

        weights = torch.matmul(Q, K.permute(0, 1, 3, 2))/ self.scale
        if mask is not None:
            weights = weights.masked_fill(mask == 0, -1e10)
        attention = torch.softmax(weights, dim=-1)
        x = torch.matmul(self.dropout(attention), V)
        x = x.permute(0, 2, 1, 3).contiguous()
        x = x.view(batch_size, -1, self.hid_dim)
        x = self.Z(x)
        return x # [batch_size, seq_len, hid_dim]