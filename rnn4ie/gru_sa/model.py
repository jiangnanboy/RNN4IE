import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from rnn4ie.util import crf

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class GRUSAIE(nn.Module):
    def __init__(self, num_embeddings, embedding_dim,
                 hid_dim, num_layers, bidirectional,
                 num_class, dropout, PAD_IDX, use_crf=True):
        super(GRUSAIE, self).__init__()
        # Hyper-parameters
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.hid_dim = hid_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.num_class = num_class
        self.dropout = dropout
        self.use_crf = use_crf
        # Components
        self.embedding = nn.Embedding(num_embeddings, embedding_dim, padding_idx=PAD_IDX)
        self.bigru = nn.GRU(embedding_dim, hid_dim, num_layers,
                            batch_first=True,
                            bidirectional=bidirectional, dropout=dropout)
        fc_in_dim = hid_dim * 2 if bidirectional else hid_dim
        self.unigru = nn.GRU(fc_in_dim, fc_in_dim, num_layers=1, batch_first=True, bidirectional=False, dropout=dropout)

        self.fc_attn = nn.Linear(fc_in_dim, fc_in_dim)
        self.fc_attn_fc = nn.Linear(fc_in_dim, fc_in_dim)
        self.fc_ht = nn.Linear(fc_in_dim, fc_in_dim)

        self.fc_out = nn.Linear(fc_in_dim, num_class)
        if use_crf:
            self.crf = crf.CRF(num_class, batch_first=True)

    def forward(self, padded_input, input_lengths):
        '''
        :param padded_input: [batch_size, seq_len]
        :param input_lengths: [batch_size]
        :return:
        '''
        batch_size, seq_len = padded_input.shape
        padded_input = self.embedding(padded_input)  # [batch_size, seq_len, embedding_dim]
        # GRU Layers
        total_length = padded_input.size(1)  # get the max sequence length
        packed_input = pack_padded_sequence(padded_input, input_lengths.cpu(), batch_first=True)
        packed_output, bigru_hidden = self.bigru(packed_input)
        bigru_hidden_cat = torch.cat([bigru_hidden[-1], bigru_hidden[-2]], dim=1)  # [batch_size, hid_dim*2]
        bigru_output, _ = pad_packed_sequence(packed_output,
                                              batch_first=True,
                                              total_length=total_length)  # [batch_size, seq_len, hid_dim*2]
        unigru_output, _ = self.unigru(bigru_output)  # [batch_size, seq_len, hid_dim*2]
        outputs = torch.zeros(batch_size, seq_len, self.num_class).to(DEVICE)
        for i in range(0, seq_len):
            if i is 0:
                unigru_cell_output = bigru_hidden_cat
            else:
                unigru_cell_output = unigru_output[:, i - 1, :]  # [batch_size, hid_dim*2]
            unigru_cell_output = unigru_cell_output.unsqueeze(1)  # [batch_size, 1, hid_dim*2]
            attn_energies = torch.sum(unigru_cell_output * bigru_output, dim=2)  # [batch_size, seq_len]
            att_weights = F.softmax(attn_energies, dim=1).unsqueeze(1)  # [batch_size, 1, seq_len]
            context = att_weights.bmm(bigru_output)  # [batch_size, 1, hid_dim*2]
            context = context.squeeze(1)
            bigru_cell_output = bigru_output[:, i, :]  # [batch_size, hid_dim*2]
            context_fc = self.fc_attn(context)
            context_fc_fc = self.fc_attn_fc(context_fc)
            ht_fc = self.fc_ht(bigru_cell_output)
            ft = context_fc * torch.sigmoid(context_fc_fc + ht_fc) + bigru_cell_output  # [batch_size, hid_dim*2]
            out = self.fc_out(ft)  # [batch_size, num_classes]
            outputs[:, i, :] = out

        if self.use_crf:
            outputs = self.crf.decode(outputs)
        return outputs  # [batch_size, seq_len, num_class]

    def log_likelihood(self, source, target, input_lengths):
        '''
        :param source: [batch_size, src_len]
        :param target: [batch_size, src_len]
        :param input_lengths: [batch_size]
        :return:
        '''
        batch_size, seq_len = source.shape
        padded_input = self.embedding(source)  # [batch_size, seq_len, embedding_dim]
        # GRU Layers
        total_length = padded_input.size(1)  # get the max sequence length
        packed_input = pack_padded_sequence(padded_input, input_lengths.cpu(), batch_first=True)
        packed_output, bigru_hidden = self.bigru(packed_input)
        bigru_hidden_cat = torch.cat([bigru_hidden[-1], bigru_hidden[-2]], dim=1)  # [batch_size, hid_dim*2]
        bigru_output, _ = pad_packed_sequence(packed_output,
                                              batch_first=True,
                                              total_length=total_length)  # [batch_size, seq_len, hid_dim*2]
        unigru_output, _ = self.unigru(bigru_output)  # [batch_size, seq_len, hid_dim*2]
        outputs = torch.zeros(batch_size, seq_len, self.num_class).to(DEVICE)
        for i in range(0, seq_len):
            if i is 0:
                unigru_cell_output = bigru_hidden_cat
            else:
                unigru_cell_output = unigru_output[:, i - 1, :]  # [batch_size, hid_dim*2]
            unigru_cell_output = unigru_cell_output.unsqueeze(1)  # [batch_size, 1, hid_dim*2]
            attn_energies = torch.sum(unigru_cell_output * bigru_output, dim=2)  # [batch_size, seq_len]
            att_weights = F.softmax(attn_energies, dim=1).unsqueeze(1)  # [batch_size, 1, seq_len]
            context = att_weights.bmm(bigru_output)  # [batch_size, 1, hid_dim*2]
            context = context.squeeze(1)
            bigru_cell_output = bigru_output[:, i, :]  # [batch_size, hid_dim*2]
            context_fc = self.fc_attn(context)
            context_fc_fc = self.fc_attn_fc(context_fc)
            ht_fc = self.fc_ht(bigru_cell_output)
            ft = context_fc * torch.sigmoid(context_fc_fc + ht_fc) + bigru_cell_output  # [batch_size, hid_dim*2]
            out = self.fc_out(ft)  # [batch_size, num_classes]
            outputs[:, i, :] = out

        return -self.crf(outputs, target)
