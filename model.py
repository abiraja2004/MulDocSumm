from copy import deepcopy

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from dataloading import SOS_IDX
from utils import append, truncate, word_drop


MAXLEN = 15

class SelfAttnCVAE(nn.Module):
    def __init__(self, var_encoder, prior_net, decoder, z_attention, c_attention):
        super().__init__()
        self.var_encoder = var_encoder # shared across docs and summ
        self.prior_net = prior_net
        self.c_attention = c_attention
        self.z_attention = z_attention # take num_encoders into acount
        self.decoder = decoder

    def _reparameterize(self, mu, log_var):
        z = torch.rand_like(mu) * (log_var/2).exp() + mu
        return z.unsqueeze(1) # (B, 1, 1100)

    def forward(self, batch):
        # encoders
        docs = filter(lambda x: 'doc' in x[0], zip(batch.fields, batch))
        variational_params = {}
        x_fixed_enc = {}
        for field, doc in docs:
            variational_params[field], x_fixed_enc[field] =\
                self.var_encoder(doc, batch.summ)
        # reparameterize
        zs = {}
        for field, (mu, log_var) in variational_params.items():
            zs[field] = self._reparameterize(mu, log_var)
        # prior network
        prior_params = {}
        for filed, doc in docs:
            prior_params[field] = self.prior_net(doc)
        # self-attention to get Z
        large_z = self.z_attention(zs)
        # self-attention to get c
        context = self.c_attention(x_fixed_enc)
        # decoder with  c and Z
        recon_logits = self.decoder(batch.summ, large_z, context)
        return variational_params, prior_params, recon_logits

class FixedEncoder(nn.Module):
    def __init__(self, vocab_size, hidden_dim=600, bidrectional=True):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, 300)
        self.lstm = nn.LSTM(300, hidden_dim, batch_first=True,
                            bidirectional=bidrectional)

    def encode(self, sents):
        sents, lengths = sents
        embedded = self.embedding(sents) # (B, l, 300)
        packed = pack_padded_sequence(embedded, lengths, batch_first=True)
        _, last_hidden = self.lstm(packed) # (2*B, hidden_dim)
        fixed_enc = torch.cat([last_hidden[0], last_hidden[1]], dim=-1)
        return fixed_enc # (B, hidden_dim*2)

class VariationalEncoder(FixedEncoder):
    def __init__(self, latent_dim, vocab_size, hidden_dim, bidirectional):
        super().__init__(vocab_size, hidden_dim, bidirectional)
        hidden_dim *= 2
        hidden_dim = hidden_dim * 2 if bidirectional else hidden_dim
        self.mu_linear = nn.Linear(hidden_dim, latent_dim)
        self.var_linear = nn.Linear(hidden_dim, latent_dim)

    def forward(self, x, y):
        x_fixed_enc = self.encode(x)
        y_fixed_enc = self.encode(y)
        fixed_enc = torch.cat([x_fixed_enc, y_fixed_enc], dim=-1)
        mu = self.mu_linear(fixed_enc) # (B, latent_dim)
        log_var = self.var_linear(fixed_enc) # (B, latent_dim)
        return (mu, log_var), x_fixed_enc

# tie lstm weights with variational encoder
class PriorNetwork(FixedEncoder):
    def __init__(self, latent_dim, vocab_size, hidden_dim, bidirectional):
        super().__init__(vocab_size, hidden_dim, bidirectional)
        hidden_dim = hidden_dim * 2 if bidirectional else hidden_dim
        self.mu_linear = nn.Linear(hidden_dim, latent_dim)
        self.var_linear = nn.Linear(hidden_dim, latent_dim)

    def forward(self, x):
        fixed_enc = self.encode(x)
        mu = self.mu_linear(fixed_enc) # (B, latent_dim)
        log_var = self.var_linear(fixed_enc) # (B, latent_dim)
        return mu, log_var

class Decoder(nn.Module):
    def __init__(self, context_size, large_z_size, vocab_size,
                 hidden_dim=600):
                 #word_drop=0.1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, 300)
        self.linear = nn.Linear(large_z_size, hidden_dim)
        self.lstm = nn.LSTM(300 + context_size, hidden_dim, batch_first=True,
                                 num_layers=2)
        self.out = nn.Linear(hidden_dim, vocab_size)
        #self.word_drop = word_drop

    def _transform_hidden(self, large_z):
        h_0 = self.linear(large_z)
        c_0 = torch.zeros_like(h_0)
        return h_0, c_0

    def forward(self, y, large_z, context): # train time
        y, lengths = y
        y, _ = append(truncate(y, 'eos'), 'sos')
        embedded = self.embedding(y) # (B, l, 300)
        embedded = torch.cat([embedded, context.repeat(1, embedded.size(1), 1)])
        packed = pack_padded_sequence(embedded, lengths, batch_first=True)
        init_hidden = self._transform_hidden(large_z)
        output, _ = self.lstm(packed, init_hidden)
        #if self.word_drop > 0.:
        #    para = word_drop(para, self.word_drop) # from Bowman's paper
        recon_logits = self.out(output)
        return recon_logits  # (B, L, vocab_size)

    #def inference(self, orig, z):
    #    orig, orig_lengths = orig # (B, l), (B,)
    #    orig = self.embedding(orig) # (B, l, 300)
    #    orig_packed = pack_padded_sequence(orig, orig_lengths,
    #                                       batch_first=True)
    #    _, orig_hidden = self.lstm_orig(orig_packed)
    #    y = []
    #    B = orig.size(0)
    #    input_ = torch.full((B,1), SOS_IDX, device=orig.device,
    #                        dtype=torch.long)
    #    hidden = orig_hidden
    #    for t in range(MAXLEN):
    #        input_ = self.embedding(input_) # (B, 1, 300)
    #        input_ = torch.cat([input_, z], dim=-1) # z (B, 1, 1100)
    #        output, hidden = self.lstm_para(input_, hidden)
    #        output = self.linear(output) # (B, 1, vocab_size)
    #        _, topi = output.topk(1) # (B, 1, 1)
    #        input_ = topi.squeeze(1)
    #        y.append(input_) # list of (B, 1)
    #    return torch.cat(y, dim=-1) # (B, L)










class Encoder(nn.Module):
    def __init__(self, vocab_size, hidden_dim=600, latent_dim=100,
                 bidrectional=True):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, 300)
        self.lstm = nn.LSTM(300, hidden_dim, batch_first=True,
                            bidirectional=bidrectional)
        hidden_dim = hidden_dim * 2 if bidrectional else hidden_dim
        self.mu_linear = nn.Linear(hidden_dim, latent_dim)
        self.logvar_linear = nn.Linear(hidden_dim, latent_dim)

    def forward(self, sents):
        sents, lengths = sents
        embedded = self.embedding(sents) # (B, l, 300)
        packed = pack_padded_sequence(embedded, lengths, batch_first=True)
        _, last_hidden = self.lstm_orig(packed) # (2*B, hidden_dim)
        last_hidden = torch.cat([last_hidden[0], last_hidden[1]], dim=-1) # (B, hidden_dim*2)
        mu, log_var = self.mu_linear(last_hidden), self.logvar_linear(last_hidden)
        return mu, log_var # (B, latent_dim), (B, latent_dim)


class CVAE(nn.Module):
    def __init__(self, encoder, decoder, vocab_size=None, hidden_dim=600,
                 latent_dim=1100):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        # TODO: activation?
        self.mu_linear = nn.Linear(hidden_dim, latent_dim)
        self.var_linear = nn.Linear(hidden_dim, latent_dim)
        self.latent_dim = latent_dim

    def _reparameterize(self, mu, log_var):
        z = torch.rand_like(mu) * (log_var/2).exp() + mu
        return z.unsqueeze(1) # (B, 1, 1100)

    def forward(self, orig, para=None):
        h_t = self.encoder(orig, para)
        mu = self.mu_linear(h_t) # (B, latent_dim)
        log_var = self.var_linear(h_t) # (B, latent_dim)
        z = self._reparameterize(mu, log_var)
        logits = self.decoder(orig, para, z)
        return logits, mu, log_var # ((B, L, vocab_size), (B, 1100), (B, 1100)

    def inference(self, orig):
        B = orig[0].size(0)
        z = torch.randn(B, 1, self.latent_dim, device= orig[0].device) # sample from prior
        generated = self.decoder.inference(orig, z)
        return generated # (B, MAXLEN)


class Encoder(nn.Module):
    def __init__(self, vocab_size, hidden_dim=600):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, 300)
        self.lstm_orig = nn.LSTM(300, hidden_dim, batch_first=True)
        self.lstm_para = nn.LSTM(300, hidden_dim, batch_first=True)

    def forward(self, orig, para):
        orig, orig_lengths = orig # (B, l), (B,)
        para, para_lengths = para
        orig = self.embedding(orig) # (B, l, 300)
        para = self.embedding(para)
        orig_packed = pack_padded_sequence(orig, orig_lengths,
                                           batch_first=True)
        # TODO: try parallel encoding w/o dependencies
        _, orig_hidden = self.lstm_orig(orig_packed)
        # no packing due to paired input
        para_output, _ = self.lstm_para(para, orig_hidden)
        B = para.size(0)
        h_t = para_output[range(B), para_lengths-1]
        return h_t # (B, hidden_dim)


class Decoder(nn.Module):
    def __init__(self, vocab_size, hidden_dim=600, latent_dim=1100,
                 word_drop=0.1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, 300)
        self.lstm_orig = nn.LSTM(300, hidden_dim, batch_first=True,
                                 num_layers=2)
        self.lstm_para = nn.LSTM(300 + latent_dim, hidden_dim,
                                 batch_first=True, num_layers=2)
        self.linear = nn.Linear(hidden_dim, vocab_size)
        self.word_drop = word_drop

    def forward(self, orig, para, z): # train time
        orig, orig_lengths = orig # (B, l), (B,)
        orig = self.embedding(orig) # (B, l, 300)
        orig_packed = pack_padded_sequence(orig, orig_lengths,
                                           batch_first=True)
        _, orig_hidden = self.lstm_orig(orig_packed)
        para, _ = append(truncate(para, 'eos'), 'sos')
        if self.word_drop > 0.:
            para = word_drop(para, self.word_drop) # from Bowman's paper
        para = self.embedding(para)
        L = para.size(1)
        para_z = torch.cat([para, z.repeat(1, L, 1)], dim=-1) # (B, L, 1100+300)
        para_output, _ = self.lstm_para(para_z, orig_hidden) # no packing
        logits = self.linear(para_output)
        return logits # (B, L, vocab_size)

    def inference(self, orig, z):
        orig, orig_lengths = orig # (B, l), (B,)
        orig = self.embedding(orig) # (B, l, 300)
        orig_packed = pack_padded_sequence(orig, orig_lengths,
                                           batch_first=True)
        _, orig_hidden = self.lstm_orig(orig_packed)
        y = []
        B = orig.size(0)
        input_ = torch.full((B,1), SOS_IDX, device=orig.device,
                            dtype=torch.long)
        hidden = orig_hidden
        for t in range(MAXLEN):
            input_ = self.embedding(input_) # (B, 1, 300)
            input_ = torch.cat([input_, z], dim=-1) # z (B, 1, 1100)
            output, hidden = self.lstm_para(input_, hidden)
            output = self.linear(output) # (B, 1, vocab_size)
            _, topi = output.topk(1) # (B, 1, 1)
            input_ = topi.squeeze(1)
            y.append(input_) # list of (B, 1)
        return torch.cat(y, dim=-1) # (B, L)


#def build_SelfAttnCVAE(vocab_size, hidden_dim, latent_dim, word_drop, bow_loss,
#                  share_emb=False, share_orig_enc=False, device=None):
# TODO: share fixed encoder weight
# TODO: implement attentions and how to deal with its dimension?
def build_SelfAttnCVAE(vocab_size, hidden_dim, latent_dim, bidirectional,
                       context_size, large_z_size,
                       share_emb=False, share_orig_enc=False, device=None):
    var_encoder = VariationalEncoder(latent_dim, vocab_size, hidden_dim,
                                     bidirectional=bidirectional)
    prior_net = PriorNetwork(latent_dim, vocab_size, hidden_dim, bidirectional)
    decoder = Decoder(context_size, large_z_size, vocab_size, hidden_dim)
    if share_emb:
        decoder.embedding.weight = encoder.embedding.weight
    else:
        model = SelfAttnCVAE(encoder, decoder, vocab_size, hidden_dim, latent_dim)
    return model.to(device)


if __name__ == '__main__':

    from dataloading import MulSumData

    PATH = '~/hwijeen/MulDocSumm/data'
    FILE = 'rottentomatoes_prepared'

    data = MulSumData(PATH, FILE, 5, torch.device('cuda'))
    model = build_
    for batch in data.train_iter:
        pass
