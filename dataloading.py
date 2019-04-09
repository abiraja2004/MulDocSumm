import os
import torch
import logging
from torchtext.data import Field, TabularDataset, BucketIterator


# TODO: MAXLEN
MAXLEN = 30
logger = logging.getLogger(__name__)

UNK_IDX = 0
PAD_IDX = 1
SOS_IDX = 2
EOS_IDX = 3


class MulSumData(object):
    def __init__(self, data_dir, file, n, device):
        self.name = file # idebate or rottentomatoes
        self.n = n
        self.device = device
        self.data_path =  os.path.join(data_dir, file + '.txt')
        self.build()

    def build(self):
        self.DOCS, self.SUMM = self.build_field(maxlen=MAXLEN)
        self.train, self.valid, self.test =\
            self.build_dataset(self.DOCS, self.SUMM)
        self.vocab = self.build_vocab(self.DOCS, self.SUMM)
        self.train_iter, self.valid_iter, self.test_iter =\
            self.build_iterator(self.train, self.valid, self.test)
        logger.info('data size... {} / {} / {}'.format(len(self.train),
                                                       len(self.valid),
                                                       len(self.test)))
        logger.info('vocab size... {}'.format(len(self.vocab)))

    def build_field(self, maxlen=None):
        DOCS = [Field(include_lengths=True, batch_first=True,
                        preprocessing=lambda x: x[:maxlen+1],
                        eos_token='<eos>')
                for _ in range(self.n)] # list
        SUMM = Field(include_lengths=True, batch_first=True,
                        preprocessing=lambda x: x[:maxlen+1],
                        eos_token='<eos>')
        return DOCS, SUMM

    def build_dataset(self, DOCS, SUMM):
        fields = [('doc{}'.format(i), DOCS[i]) for i in range(self.n)]
        fields += [('summ', SUMM)]
        data = TabularDataset(path=self.data_path, format='tsv',
                                   fields=fields)
        train, test, valid = data.split(split_ratio=[0.8, 0.1, 0.1])
        return train, valid, test

    def build_vocab(self, DOCS, SUMM):
        # not using pretrained word vectors
        sources = []
        for data in [self.train, self.valid]:
            sources += [getattr(data, 'doc{}'.format(i))
                       for i in range(self.n)]
            sources += [getattr(data, 'summ')]
        SUMM.build_vocab(sources, max_size=30000)
        SUMM.vocab.itos.insert(2, '<sos>')
        from collections import defaultdict
        stoi = defaultdict(lambda: 0)
        stoi.update({tok: i for i, tok in enumerate(SUMM.vocab.itos)})
        SUMM.vocab.stoi = stoi
        for doc in DOCS:
            doc.vocab = SUMM.vocab
        return SUMM.vocab

    def build_iterator(self, train, valid, test):
        train_iter, valid_iter, test_iter = \
        BucketIterator.splits((train, valid, test), batch_size=32,
                              #sort_key=lambda x: (len(x.orig), len(x.para)),
                              sort_key=lambda x: len(x.summ),
                              sort_within_batch=True, repeat=False,
                              device=self.device)
        return train_iter, valid_iter, test_iter

if __name__ == "__main__":
    import torch
    from utils import reverse

    PATH = '~/hwijeen/MulDocSumm/data'
    FILE = 'rottentomatoes_prepared'

    data = MulSumData(PATH, FILE, 5, torch.device('cuda'))

    print(len(data.train_iter)) # only 94
    print(len(data.valid_iter)) # only 12
    print(len(data.test_iter)) # only 12

    for batch in data.train_iter:
        print(reverse(batch.doc0[0], data.vocab))
        print(reverse(batch.doc1[0], data.vocab))
        print(reverse(batch.doc2[0], data.vocab))
        print(reverse(batch.doc3[0], data.vocab))
        print(reverse(batch.doc4[0], data.vocab))
        print(reverse(batch.summ[0], data.vocab))
        input()

