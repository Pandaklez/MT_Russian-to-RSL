import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class TranslatorEncoder(nn.Module):
    def __init__(self, rus_num_emb, rus_emb_size, rnn_size):
        super(TranslatorEncoder, self).__init__()

        self.rus_embeddings = nn.Embedding(rus_num_emb, rus_emb_size, max_norm=1.)
        self.rus_birnn = nn.GRU(rus_emb_size, rnn_size, bidirectional=True, batch_first=True)
        self.rus_birnn.flatten_parameters()

    def forward(self, rus_sentances):
        self.rus_birnn.flatten_parameters()
        rus_embedded = self.rus_embeddings(rus_sentances)

        # rus_rnn_h.shape = (rnn_size, batch_size, rus_emb_size)
        rus_birnn_out, rus_rnn_h = self.rus_birnn(rus_embedded)
        # permute to (batch_size, rnn_size, rus_emb_size)
        rus_rnn_h = rus_rnn_h.permute(1, 0, 2)
        rus_rnn_h = rus_rnn_h.contiguous().view(rus_rnn_h.size(0), -1)

        return rus_birnn_out, rus_rnn_h


def verbose_attention(encoder_state_vectors, query_vector):
    """A descriptive version of the neural attention mechanism

    Args:
        encoder_state_vectors (torch.Tensor): 3dim tensor from bi-GRU in encoder
        query_vector (torch.Tensor): hidden state in decoder GRU
    Returns:

    """
    batch_size, num_vectors, vector_size = encoder_state_vectors.size()
    vector_scores = torch.sum(encoder_state_vectors * query_vector.view(batch_size, 1, vector_size),
                              dim=2)
    vector_probabilities = F.softmax(vector_scores, dim=1)
    weighted_vectors = encoder_state_vectors * vector_probabilities.view(batch_size, num_vectors, 1)
    context_vectors = torch.sum(weighted_vectors, dim=1)
    return context_vectors, vector_probabilities, vector_scores


def terse_attention(encoder_state_vectors, query_vector):
    """A shorter and more optimized version of the neural attention mechanism

    Args:
        encoder_state_vectors (torch.Tensor): 3dim tensor from bi-GRU in encoder
        query_vector (torch.Tensor): hidden state
    """
    vector_scores = torch.matmul(encoder_state_vectors, query_vector.unsqueeze(dim=2)).squeeze()
    vector_probabilities = F.softmax(vector_scores, dim=-1)
    context_vectors = torch.matmul(encoder_state_vectors.transpose(-2, -1),
                                   vector_probabilities.unsqueeze(dim=2)).squeeze()
    return context_vectors, vector_probabilities


class TranslatorDecoder(nn.Module):
    def __init__(self, rsl_num_emb, rsl_emb_size, rnn_size, bos_ind):
        super(TranslatorDecoder, self).__init__()

        self._rnn_size = rnn_size
        
        self.rsl_embedding = nn.Embedding(rsl_num_emb, rsl_emb_size, max_norm=1.)
        
        self.gru_cell = nn.GRUCell(rsl_emb_size + rnn_size, rnn_size)

        self.linear_map = nn.Linear(rnn_size, rnn_size)
        self.classifier = nn.Linear(rnn_size*2, rsl_num_emb)

        self.bos_ind = bos_ind
        self._sampling_temperature = 2.999

        self._cached_p_attn = []
        self._cached_ht = []
        self._cached_decoder_state = None

    def _init_indices(self, batch_size):
        return torch.ones(batch_size, dtype=torch.int64) * self.bos_ind

    def _init_context_vectors(self, batch_size):
        return torch.zeros(batch_size, self._rnn_size)

    def forward(self, encoder_state, init_h_state, rsl_sentence, sample_probability=0.0):
        #encoder_state = torch.flatten(encoder_state, start_dim=1, end_dim=-1)
        init_h_state = torch.flatten(init_h_state, start_dim=1, end_dim=-1)
        rsl_sentence = torch.flatten(rsl_sentence, start_dim=1, end_dim=-1)
        
        batch_size = encoder_state.size(0)
        output_sequence_size = 0

        if rsl_sentence is None:
            sample_probability = 1.0
        else:
            rsl_sentence = rsl_sentence.permute(1, 0)
            output_sequence_size = rsl_sentence.size(0)

        context_vectors = self._init_context_vectors(batch_size)
        h_t = self.linear_map(init_h_state)
        y_t_index = self._init_indices(batch_size)

        h_t = h_t.to(encoder_state.device)
        y_t_index = y_t_index.to(encoder_state.device)
        context_vectors = context_vectors.to(encoder_state.device)

        output_vectors = []
        self._cached_p_attn = []
        self._cached_ht = []
        self._cached_decoder_state = encoder_state.cpu().detach().numpy()

        output_sequence_size = rsl_sentence.size(0)
        for i in range(output_sequence_size):
            use_sample = np.random.random() < sample_probability
            if not use_sample:
                y_t_index = rsl_sentence[i]

            y_input_vector = self.rsl_embedding(y_t_index)
            rnn_input = torch.cat([y_input_vector, context_vectors], dim=1)

            h_t = self.gru_cell(rnn_input, h_t)
            self._cached_ht.append(h_t.cpu().detach().numpy())

            # Use the current hidden to attend to the encoder state
            context_vectors, p_attn = terse_attention(encoder_state_vectors=encoder_state, query_vector=h_t)

            self._cached_p_attn.append(p_attn.cpu().detach().numpy())

            prediction_vector = torch.cat((context_vectors, h_t), dim=1)
            score_for_y_t_index = self.classifier(F.dropout(prediction_vector, 0.1))

            if use_sample:
                p_y_t_index = F.softmax(score_for_y_t_index * self._sampling_temperature, dim=1)
                y_t_index = torch.multinomial(p_y_t_index, 1).squeeze()

            output_vectors.append(score_for_y_t_index)

        output_vectors = torch.stack(output_vectors).permute(1, 0, 2)

        return output_vectors


class Translator(nn.Module):
    def __init__(self, rus_vocab_size, rus_embedding_size,
                 rsl_vocab_size, rsl_embbeding_size, encoding_size, rsl_bos_ind):
        super(Translator, self).__init__()

        self.encoder = TranslatorEncoder(rus_vocab_size, rus_embedding_size, encoding_size)

        self.decoder = TranslatorDecoder(rsl_vocab_size, rsl_embbeding_size, encoding_size*2, rsl_bos_ind)

    def forward(self, rus_sentances, rsl_sentence, sample_probability=0.0):
        encoder_state, final_hidden_states = self.encoder(rus_sentances)
        decoded_states = self.decoder(encoder_state, final_hidden_states, rsl_sentence, sample_probability)

        return decoded_states

