#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 18 15:34:46 2023

@author: giuse
"""

import sys

def preprocess_graph(dataset):



    f_train_in, f_train_out, f_prop = 'data/'+dataset+'/train.tsv', 'data/'+dataset+'/pykeen_train.tsv', 'data/'+dataset+'/prop.tsv'
    f_test_in, f_test_out = 'data/'+dataset+'/test.tsv', 'data/'+dataset+'/pykeen_test.tsv'
    f_map_ent, f_map_rels = 'data/'+dataset+'/mapping_entities.tsv', 'data/'+dataset+'/mapping_relations.tsv'

    map_ent, map_rels = {}, {}

    with open(f_map_ent, 'r', encoding='utf-8') as fin:
        for line in fin:
            id, ent = line.strip().split('\t')
            map_ent[id] = ent

    with open(f_map_rels, 'r', encoding='utf-8') as fin:
        for line in fin:
            id, rel = line.strip().split('\t')
            map_rels[id] = rel

    with open(f_train_in, 'r', encoding='utf-8') as fin:
        with open(f_train_out, 'w', encoding='utf-8') as fout:
            for line in fin:
                h, t, r = line.strip().split('\t')
                outline = map_ent[h]+'\t'+map_rels[r]+'\t'+map_ent[t]+'\n'
                fout.write(outline)
                
    with open(f_prop, 'r', encoding='utf-8') as fin:
        with open(f_train_out, 'a', encoding='utf-8') as fout:
            for line in fin:
                h, t, r = line.strip().split('\t')
                outline = map_ent[h]+'\t'+map_rels[r]+'\t'+map_ent[t]+'\n'
                fout.write(outline)

    with open(f_test_in, 'r', encoding='utf-8') as fin:
        with open(f_test_out, 'w', encoding='utf-8') as fout:
            for line in fin:
                h, t, r = line.strip().split('\t')
                outline = map_ent[h]+'\t'+map_rels[r]+'\t'+map_ent[t]+'\n'
                fout.write(outline)
            


if __name__ == '__main__':
    args = sys.argv[1:]
    preprocess_graph(args[0])