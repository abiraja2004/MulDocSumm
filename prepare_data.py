import json
import random
import argparse


def prepare_tsv(args, data):
    """Select n docs uniformly per summary"""
    selected_data = []
    for d in data:
        if 'rottentomatoes' in args.file:
            docs = list(d['_critics'].values())
        elif 'idebate' in args.file:
            docs = list(d['_argument_sentences'].values())
        # TODO: Idebate - small data
        assert len(docs) > args.n, 'docs with number smaller than n'
        selected_docs = random.sample(docs, args.n) # list of str

        if 'rottentomatoes' in args.file:
            summ = d['_critic_consensus'] # str
        elif 'idebate' in args.file:
            summ = d['_claim'] # str

        selected_d = selected_docs + [summ] # list of list, len args.n + 1
        selected_data.append(selected_d)
    return selected_data

def write_tsv(OUT, selected_data):
    with open(OUT, 'w') as f:
        for d in selected_data:
            d_tsv = '\t'.join(d)
            f.write(d_tsv + '\n')

def prepare_json(args, data):
    """Just reformat json file for compatiability with torchtext"""
    ex_list = []
    for d in data:
        ex = {}
        if 'rottentomatoes' in args.file:
            docs = list(d['_critics'].values())
        elif 'idebate' in args.file:
            docs = list(d['_argument_sentences'].values())
        # TODO: Idebate - small data
        for i in range(len(docs)):
            ex['doc{}'.format(i+1)] = docs[i]

        if 'rottentomatoes' in args.file:
            summ = d['_critic_consensus'] # str
        elif 'idebate' in args.file:
            summ = d['_claim'] # str
        ex['summ'] = summ
        ex_list.append(ex)
    return ex_list # list of dict

def write_json(OUT, ex_list):
    with open(OUT, 'w') as f:
        for ex in ex_list:
            f.write(json.dumps(ex) + '\n')

if  __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--file", type=str, default='rottentomatoes')
    parser.add_argument("--n", type=int, default='5')
    parser.add_argument("--format", type=str, default='json')
    args = parser.parse_args()

    if 'rottentomatoes' in args.file:
        IN = 'data/rottentomatoes.json'
        OUT = 'data/rottentomatoes_prepared.{}'.format(args.format)
    elif 'idebate' in args.file:
        IN = 'data/idebate.json'
        OUT = 'data/idebate_prepared.{}'.format(args.format)

    with open(IN, 'r') as f:
        data = json.load(f) # list of dict

    if args.format == 'tsv':
        selected_data = prepare_tsv(args, data)
        write_tsv(OUT, selected_data)
    elif args.format == 'json':
        ex_list = prepare_json(args, data)
        write_json(OUT, ex_list)

    print('============ {} data prepared =========='.format(OUT))

