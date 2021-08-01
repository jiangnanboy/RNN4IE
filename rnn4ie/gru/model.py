import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from rnn4ie.util import crf

class GRUIE(nn.Module):
    def __init__(self, num_embeddings, embedding_dim,
                 hid_dim, num_layers, bidirectional,
                 num_class, dropout, PAD_IDX, use_crf=True):
        super(GRUIE, self).__init__()
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
        self.gru = nn.GRU(embedding_dim, hid_dim, num_layers,
                            batch_first=True,
                            bidirectional=bidirectional, dropout=dropout)
        fc_in_dim = hid_dim * 2 if bidirectional else hid_dim
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
        # Output Layer
        output = self.fc(output)
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
        # Output Layer
        output = self.fc(output)

        return -self.crf(output, target)
