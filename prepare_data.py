import json
import random
import argparse


if  __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--file", type=str, default='rottentomatoes')
    parser.add_argument("--n", type=int, default='5')
    args = parser.parse_args()

    if 'rottentomatoes' in args.file:
        FILE = 'data/rottentomatoes.json'
        OUT = 'data/rottentomatoes_prepared.txt'
    elif 'idebate' in args.file:
        FILE = 'data/idebate.json'
        OUT = 'data/idebate_prepared.txt'

    selected_data = []
    with open(FILE, 'r') as f:
        data = json.load(f) # list of dict

    for d in data:
        if 'rottentomatoes' in args.file:
            docs = list(d['_critics'].values())
        elif 'idebate' in args.file:
            docs = list(d['_argument_sentences'].values())
        assert len(docs) > args.n, 'docs with number smaller than n'
        selected_docs = random.sample(docs, args.n) # list of str

        if 'rottentomatoes' in args.file:
            summ = d['_critic_consensus'] # str
        elif 'idebate' in args.file:
            summ = d['_claim'] # str

        selected_d = selected_docs + [summ] # list of list, len args.n + 1
        selected_data.append(selected_d)

    with open(OUT, 'w') as f:
        for d in selected_data:
            d_tsv = '\t'.join(d)
            f.write(d_tsv + '\n')

    msg = '============ {} data prepared =========='.format(OUT)
    print(msg)
    print('\tnumer of data: {}, number of docs per summary: {}'.format(len(data), args.n))
    print('='*len(msg))

