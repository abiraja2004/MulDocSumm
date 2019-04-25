import os
import io
import torch
import logging

from torchtext.data import Field, TabularDataset, BucketIterator, Example, Dataset


# TODO: MAXLEN
MAXLEN = 30
logger = logging.getLogger(__name__)

UNK_IDX = 0
PAD_IDX = 1
SOS_IDX = 2
EOS_IDX = 3

# Modified so that fields not present in input data are left "".
class Example(Example):
    """Holds fields of arbitrary length"""
    @classmethod
    def fromdict(cls, data, fields):
        ex = cls()
        for key, vals in fields.items():
            if key not in data:
                #raise ValueError("Specified key {} was not found in "
                #                 "the input data".format(key))
                # override
                setattr(ex, key, "")
                continue
            if vals is not None:
                if not isinstance(vals, list):
                    vals = [vals]
                for val in vals:
                    name, field = val
                    setattr(ex, name, field.preprocess(data[key]))
        return ex

# Modified to use modified Example
class TabularDataset(Dataset):
    """Defines a Dataset of columns stored in CSV, TSV, or JSON format."""

    def __init__(self, path, format, fields, skip_header=False,
                 csv_reader_params={}, **kwargs):
        """Create a TabularDataset given a path, file format, and field list.

        Arguments:
            path (str): Path to the data file.
            format (str): The format of the data file. One of "CSV", "TSV", or
                "JSON" (case-insensitive).
            fields (list(tuple(str, Field)) or dict[str: tuple(str, Field)]:
                If using a list, the format must be CSV or TSV, and the values of the list
                should be tuples of (name, field).
                The fields should be in the same order as the columns in the CSV or TSV
                file, while tuples of (name, None) represent columns that will be ignored.

                If using a dict, the keys should be a subset of the JSON keys or CSV/TSV
                columns, and the values should be tuples of (name, field).
                Keys not present in the input dictionary are ignored.
                This allows the user to rename columns from their JSON/CSV/TSV key names
                and also enables selecting a subset of columns to load.
            skip_header (bool): Whether to skip the first line of the input file.
            csv_reader_params(dict): Parameters to pass to the csv reader.
                Only relevant when format is csv or tsv.
                See
                https://docs.python.org/3/library/csv.html#csv.reader
                for more details.
        """
        format = format.lower()
        # override
        make_example = {
            'json': Example.fromJSON, 'dict': Example.fromdict,
            'tsv': Example.fromCSV, 'csv': Example.fromCSV}[format]

        with io.open(os.path.expanduser(path), encoding="utf8") as f:
            if format == 'csv':
                reader = unicode_csv_reader(f, **csv_reader_params)
            elif format == 'tsv':
                reader = unicode_csv_reader(f, delimiter='\t', **csv_reader_params)
            else:
                reader = f

            if format in ['csv', 'tsv'] and isinstance(fields, dict):
                if skip_header:
                    raise ValueError('When using a dict to specify fields with a {} file,'
                                     'skip_header must be False and'
                                     'the file must have a header.'.format(format))
                header = next(reader)
                field_to_index = {f: header.index(f) for f in fields.keys()}
                make_example = partial(make_example, field_to_index=field_to_index)

            if skip_header:
                next(reader)

            examples = [make_example(line, fields) for line in reader]

        if isinstance(fields, dict):
            fields, field_dict = [], fields
            for field in field_dict.values():
                if isinstance(field, list):
                    fields.extend(field)
                else:
                    fields.append(field)

        super(TabularDataset, self).__init__(examples, fields, **kwargs)


class MulSumData(object):
    def __init__(self, data_dir, file, n, device):
        self.name = file # idebate or rottentomatoes
        self.n = n
        self.device = device
        self.data_path =  os.path.join(data_dir, file + '.json')
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
                        eos_token='<eos>', lower=True, tokenize='toktok')
                for _ in range(self.n)] # list
        SUMM = Field(include_lengths=True, batch_first=True,
                        preprocessing=lambda x: x[:maxlen+1],
                        eos_token='<eos>', lower=True, is_target=True)
        return DOCS, SUMM

    def build_dataset(self, DOCS, SUMM):
        fields = {'doc{}'.format(i): ('doc{}'.format(i), f)
                  for i, f in enumerate(DOCS, 1)}
        fields['summ'] = ('summ', SUMM)
        data = TabularDataset(path=self.data_path, format='json',
                                   fields=fields)
        train, test, valid = data.split(split_ratio=[0.8, 0.1, 0.1])
        return train, valid, test

    # TODO: check total vocab size
    def build_vocab(self, DOCS, SUMM):
        # not using pretrained word vectors
        sources = []
        for data in [self.train, self.valid]:
            sources += [getattr(data, 'doc{}'.format(i))
                       for i in range(1, self.n+1)]
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

    data = MulSumData(PATH, FILE, 99, torch.device('cuda'))

    print(len(data.train_iter)) # only 94
    print(len(data.valid_iter)) # only 12
    print(len(data.test_iter)) # only 12

    for batch in data.train_iter:
        print('batch: ', batch)
        print(batch.doc1[0])
        input()
        print(reverse(batch.doc1[0], data.vocab))
        print(reverse(batch.doc2[0], data.vocab))
        print(reverse(batch.doc3[0], data.vocab))
        print(reverse(batch.doc4[0], data.vocab))
        print(reverse(batch.doc5[0], data.vocab))
        print(reverse(batch.summ[0], data.vocab))
        input()

