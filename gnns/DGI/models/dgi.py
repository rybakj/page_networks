import torch
import torch.nn as nn
from layers import GCN, AvgReadout, Discriminator

# model = DGI(ft_size, hid_units, nonlinearity)
# logits = model(features, shuf_fts, sp_adj if sparse else adj, sparse, None, None, None)
class DGI(nn.Module):
    # n_in = number of input (node) features
    # n_h = number of hidden layers (and outputs)
    def __init__(self, n_in, n_h, activation):
        super(DGI, self).__init__()
        self.gcn = GCN(n_in, n_h, activation)
        # Read function
        self.read = AvgReadout()

        self.sigm = nn.Sigmoid()

        # Discriminator
        self.disc = Discriminator(n_h)

    def forward(self, seq1, seq2, adj, sparse, msk, samp_bias1, samp_bias2):
        # seq1 = original features
        # seq2 = shuffled features


        # h1 = embeddings of seq1 nodes
        h_1 = self.gcn(seq1, adj, sparse)
        # c: summary vector (read encodings of seq1)
        # If mask is specified, c is obtained only over the unmasked embeddings (h)
        # If mask not specified, ger Read over all embeddings
        c = self.read(h_1, msk)
        # Read is bilinear, now apply sigmoid to convert to probability of sample
        # being a positive one
        c = self.sigm(c)

        h_2 = self.gcn(seq2, adj, sparse)

        ret = self.disc(c, h_1, h_2, samp_bias1, samp_bias2)
        # return logits (prob of sample being true) from h_1 (true features) and h_2 (resampled features)

        return ret

    # Detach the return variables
    def embed(self, seq, adj, sparse, msk):
        h_1 = self.gcn(seq, adj, sparse)
        c = self.read(h_1, msk)

        return h_1.detach(), c.detach()

