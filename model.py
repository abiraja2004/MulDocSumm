import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from module import get_attention
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

    def reparameterize(self, mu, log_var):
        z = torch.rand_like(mu) * (log_var/2).exp() + mu
        return z # (B, latent_dim)

    def forward(self, batch):
        # encoders
        # FIXME iterating batch when it has multiple fields
        docs = list(filter(lambda x: 'doc' in x[0], zip(batch.fields, list(iter(batch))[0])))
        variational_params = {}
        x_fixed_enc = {}
        for field, doc in docs: # convert to list for multiple iteration
            variational_params[field], x_fixed_enc[field] =\
                self.var_encoder(doc, batch.summ)
        # reparameterize
        zs = {}
        for field, (mu, log_var) in variational_params.items():
            zs[field] = self.reparameterize(mu, log_var)
        # prior network
        prior_params = {}
        for field, doc in docs:
            prior_params[field] = self.prior_net(doc)
        # self-attention to get Z
        large_z = self.z_attention(zs)
        # self-attention to get c
        context = self.c_attention(x_fixed_enc)
        # decoder with  c and Z
        recon_logits = self.decoder(batch.summ, large_z, context)
        return variational_params, prior_params, recon_logits

    def inference(self, batch):
        docs = [x for x in zip(batch.fields, list(iter(batch))[0]) if 'doc' in x[0]]
        prior_params = {}
        for field, doc in docs:
            prior_params[field] = self.prior_net(doc)
        x_fixed_enc = {}
        for field, doc in docs: # convert to list for multiple iteration
            _, x_fixed_enc[field] = self.var_encoder(doc, batch.summ)
        context = self.c_attention(x_fixed_enc)
        generated = self.decoder.inference(prior_params, context)
        return generated

class FixedEncoder(nn.Module):
    def __init__(self, vocab_size, hidden_dim=600, bidrectional=True):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, 300)
        self.lstm = nn.LSTM(300, hidden_dim, batch_first=True,
                            bidirectional=bidrectional)

    def encode(self, sents):
        sents, lengths = sents
        embedded = self.embedding(sents) # (B, l, 300)
        #packed = pack_padded_sequence(embedded, lengths, batch_first=True)
        #_, last_hidden = self.lstm(packed) # (2, B, hidden_dim)
        _, (last_hidden, _) = self.lstm(embedded) # (2, B, hidden_dim)
        fixed_enc = torch.cat([last_hidden[0], last_hidden[1]], dim=-1)
        return fixed_enc # (B, hidden_dim*2)


class VariationalEncoder(FixedEncoder):
    def __init__(self, latent_dim, vocab_size, hidden_dim, bidirectional):
        super().__init__(vocab_size, hidden_dim, bidirectional)
        hidden_dim *= 2 # concatenation of x and y
        hidden_dim = hidden_dim * 2 if bidirectional else hidden_dim
        self.mu_linear = nn.Linear(hidden_dim, latent_dim)
        self.var_linear = nn.Linear(hidden_dim, latent_dim)

    def forward(self, x, y):
        x_fixed_enc = self.encode(x)
        y_fixed_enc = self.encode(y)
        fixed_enc = torch.cat([x_fixed_enc, y_fixed_enc], dim=-1) # (B, hidden_dim*2)
        mu = self.mu_linear(fixed_enc) # (B, latent_dim)
        log_var = self.var_linear(fixed_enc) # (B, latent_dim)
        return (mu, log_var), x_fixed_enc # / / (B, hidden_dim*2)


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
    def __init__(self, enc_hidden_dim, latent_dim, vocab_size, hidden_dim=600,
                 num_layers=2):
                 #word_drop=0.1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, 300)
        self.linear = nn.Linear(latent_dim, hidden_dim)
        self.lstm = nn.LSTM(300 + enc_hidden_dim, hidden_dim, batch_first=True,
                                 num_layers=2)
        self.out = nn.Linear(hidden_dim, vocab_size)
        self.num_layers = num_layers
        #self.word_drop = word_drop

    # TODO: consider num_layers
    def _transform_hidden(self, large_z):
        # (num_layers, B, hidden_dim)
        h_0 = self.linear(large_z).transpose(0,1).repeat(self.num_layers, 1, 1)
        c_0 = torch.zeros_like(h_0)
        return h_0, c_0

    def forward(self, y, large_z, context): # train time
        y, lengths = append(truncate(y, 'eos'), 'sos')
        embedded = self.embedding(y) # (B, l, 300)
        embedded = torch.cat([embedded, context.repeat(1, embedded.size(1), 1)],
                             dim=-1)
        packed = pack_padded_sequence(embedded, lengths, batch_first=True)
        init_hidden = self._transform_hidden(large_z)
        packed_output, _ = self.lstm(packed, init_hidden)
        #if self.word_drop > 0.:
        #    para = word_drop(para, self.word_drop) # from Bowman's paper
        total_length = embedded.size(1)
        output, _ = pad_packed_sequence(packed_output, batch_first=True,
                                     total_length=total_length)
        recon_logits = self.out(output)
        return recon_logits  # (B, L, vocab_size)

    def inference(self, prior_params, context):
        # sample from prior_params

        # decode with <s> and context
        raise NotImplementedError

        #orig, orig_lengths = orig # (B, l), (B,)
        #orig = self.embedding(orig) # (B, l, 300)
        #orig_packed = pack_padded_sequence(orig, orig_lengths,
        #                                   batch_first=True)
        #_, orig_hidden = self.lstm_orig(orig_packed)
        #y = []
        #B = orig.size(0)
        #input_ = torch.full((B,1), SOS_IDX, device=orig.device,
        #                    dtype=torch.long)
        #hidden = orig_hidden
        #for t in range(MAXLEN):
        #    input_ = self.embedding(input_) # (B, 1, 300)
        #    input_ = torch.cat([input_, z], dim=-1) # z (B, 1, 1100)
        #    output, hidden = self.lstm_para(input_, hidden)
        #    output = self.linear(output) # (B, 1, vocab_size)
        #    _, topi = output.topk(1) # (B, 1, 1)
        #    input_ = topi.squeeze(1)
        #    y.append(input_) # list of (B, 1)
        #return torch.cat(y, dim=-1) # (B, L)


def build_SelfAttnCVAE(vocab_size, hidden_dim, latent_dim, enc_bidirectional,
                       attn_type='proj',
                       share_emb=True, share_fixed_enc=True, device=None):
    var_encoder = VariationalEncoder(latent_dim, vocab_size, hidden_dim,
                                     bidirectional=enc_bidirectional)
    prior_net = PriorNetwork(latent_dim, vocab_size, hidden_dim, enc_bidirectional)
    enc_hidden_dim = hidden_dim * 2 if enc_bidirectional else hidden_dim
    decoder = Decoder(enc_hidden_dim, latent_dim, vocab_size, hidden_dim)
    z_attention = get_attention(attn_type, latent_dim)
    c_attention = get_attention(attn_type, enc_hidden_dim)
    if share_emb:
        decoder.embedding.weight = prior_net.embedding.weight
    if share_fixed_enc:
        var_encoder.embedding.weight = prior_net.embedding.weight
        for var_w, prior_w in zip(var_encoder.lstm.all_weights, prior_net.lstm.all_weights):
            var_w = prior_w
    model = SelfAttnCVAE(var_encoder, prior_net, decoder, z_attention, c_attention)
    return model.to(device)


if __name__ == '__main__':

    from dataloading import MulSumData

    PATH = '~/hwijeen/MulDocSumm/data'
    FILE = 'rottentomatoes_prepared'
    DEVICE = torch.device('cuda:1')

    data = MulSumData(PATH, FILE, 5, DEVICE)
    model = build_SelfAttnCVAE(len(data.vocab), hidden_dim=600, latent_dim=300,
                               enc_bidirectional=True, device=DEVICE)
    print(model)
    for batch in data.train_iter:
        variational_params, prior_params, recon_logits = model(batch)
        break


