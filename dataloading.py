import os
import torch
import logging
from torchtext.data import Field, TabularDataset, BucketIterator


MAXLEN = 15
logger = logging.getLogger(__name__)

UNK_IDX = 0
PAD_IDX = 1
SOS_IDX = 2
EOS_IDX = 3


class Data(object):
    def __init__(self, data_dir, file, device):
        self.name = file # idebate or rottentomatoes
        self.device = device
        self.data_path =  os.path.join(data_dir, file + '.json')
        self.build()

    def build(self):
        self.DOCS, self.SUMM = self.build_field(maxlen=MAXLEN)
        logger.info('building datasets... this takes a while')
        self.train, self.val, self.test =\
            self.build_dataset(self.DOCS, self.SUMM)
        self.vocab = self.build_vocab(self.DOCS, self.SUMM,
                                      self.train.docs, self.train.summ,
                                      self.val.docs, self.val.summ)
        self.train_iter, self.valid_iter, self.test_iter =\
            self.build_iterator(self.train, self.val, self.test)
        logger.info('data size... {} / {} / {}'.format(len(self.train),
                                                       len(self.val),
                                                       len(self.test)))
        logger.info('vocab size... {}'.format(len(self.vocab)))

    def build_field(self, maxlen=None):
        DOCS = Field(include_lengths=True, batch_first=True,
                        preprocessing=lambda x: x[:maxlen+1],
                        eos_token='<eos>')
        SUMM = Field(include_lengths=True, batch_first=True,
                        preprocessing=lambda x: x[:maxlen+1],
                        eos_token='<eos>')
        return DOCS, SUMM

    def build_dataset(self, DOCS, SUMM):
        if 'rottentomatoes' in self.name:
            train_val = TabularDataset(path=self.data_path, format='json',
                                       fields={'_critics': ('docs', DOCS),
                                               '_critic_consensus': ('summ', SUMM)})
        elif 'idebate' in self.name:
            train_val = TabularDataset(path=self.data_path, format='json',
                                       fields={'_argument_sentences': ('docs', DOCS),
                                               '_claim': ('summ', SUMM)})
        train, test, val = train_val.split(split_ratio=[0.8, 0.1, 0.1])
        return train, val, test

    # TODO: add sos token
    def build_vocab(self, DOCS, SUMM, *args):
        # not using pretrained word vectors
        DOCS.build_vocab(args, max_size=30000)
        DOCS.vocab.itos.insert(2, '<sos>')
        from collections import defaultdict
        stoi = defaultdict(lambda x:0)
        stoi.update({tok: i for i, tok in enumerate(DOCS.vocab.itos)})
        DOCS.vocab.stoi = stoi
        SUMM.vocab = DOCS.vocab
        return DOCS.vocab

    def build_iterator(self, train, val, test):
        train_iter, valid_iter, test_iter = \
        BucketIterator.splits((train, val, test), batch_size=32,
                              sort_key=lambda x: (len(x.orig), len(x.para)),
                              sort_within_batch=True, repeat=False,
                              device=self.device)
        return train_iter, valid_iter, test_iter

if __name__ == "__main__":
    import torch
    
    PATH = '~/hwijeen/MulDocSumm/data'
    FILE = 'rottentomatoes_prepared'

    data = Data(PATH, FILE, torch.device('cuda'))

