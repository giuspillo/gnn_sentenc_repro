from pykeen.triples import TriplesFactory
from pykeen.pipeline import pipeline
import numpy
import pandas as pd
from sklearn.model_selection import train_test_split
import os
from os import path
import torch
from termcolor import colored
import pickle
import sys

# organized as list so that it is easy to automatically iterate 
# if you want to add other datasets, models, or embedding dimensions

def gen_embedding_dict(dataset, folder):

    map_graph = {}
    fin = open('data/'+dataset+'/mapping_entities.tsv', 'r', encoding='utf-8')
    for line in fin:
        a,b=line.strip().split('\t')
        id = int(a)
        map_graph[id]=b
    fin.close()

    matrix = numpy.loadtxt(folder+'/embeddings.tsv', delimiter='\t')
    
    map_emb = {}
    missing = set()
    fin = open(folder+'/entities_to_id.tsv', 'r', encoding='utf-8')
    for line in fin:
        name, strid = line.strip().split('\t')
        id = int(strid)
        map_emb[name] = id
    fin.close()
    
    curdict = {}
    for entity in map_graph:
        
        try:
            id_emb = map_emb[map_graph[entity]]
            emb = matrix[id_emb]
            curdict[entity]=emb
        except:
            missing.add(entity)        
    

    train_ent = set()
    test_ent = set()
    
    fin = open('data/'+dataset+'/train.tsv', 'r', encoding='utf-8')
    for line in fin:
        a,b,c = line.strip().split('\t')
        user = int(a)
        item = int(b)
    fin.close()
    
    fin = open('data/'+dataset+'/test.tsv', 'r', encoding='utf-8')
    for line in fin:
        a,b,c = line.strip().split('\t')
        user = int(a)
        item = int(b)
    fin.close()
    
    out = open(folder+'.pickle', 'wb')
    pickle.dump(curdict, out)
    out.flush()
    out.close()




def main(dataset, emb_model, emb_dim, emb_epochs):

    # output strings
    printline = dataset+' - '+emb_model+' - k='+str(emb_dim)
    print(colored('Starting ' + printline,'blue'))

    # utility folders and paths
    folder = 'data/' + dataset + '/' + dataset+'_'+emb_model+'_k='+str(emb_dim)
    print(folder)

    train_path = 'data/' + dataset + '/' + 'pykeen_train.tsv'
    test_path = 'data/' + dataset + '/' +'pykeen_test.tsv'

    checkpoint_name_file = dataset+'_'+emb_model+'_k='+str(emb_dim)

    if os.path.isfile(folder+'/embeddings.tsv'):
        if not os.path.isfile(folder+'.pickle'):
            gen_embedding_dict(dataset, folder)
        else:
            print(colored('Existing embedding dict in ' + folder + '.pickle'), 'blue')
        print(colored('Existing embedding in ' + folder,'blue'))

    else:

        try:      

            print(colored('Starting learning:' + folder,'blue'))
            print("Starting learning:", printline)
            

            emb_training = TriplesFactory.from_path(
                train_path,
                create_inverse_triples=True,
            )

            emb_testing = TriplesFactory.from_path(
                test_path,
                entity_to_id=emb_training.entity_to_id,
                relation_to_id=emb_training.relation_to_id,
                create_inverse_triples=True,
            )

            result = pipeline(
                training=emb_training,
                testing=emb_testing,
                model=emb_model,
                model_kwargs=dict(embedding_dim=int(emb_dim)),
                evaluation_fallback = True,
                training_kwargs=dict(
                    num_epochs=int(emb_epochs),
                    checkpoint_name=checkpoint_name_file,
                    checkpoint_directory="data/"+dataset+"/KGE_checkpoints/",
                    checkpoint_frequency=1
                ),
            )

            if not os.path.exists(folder):
                os.mkdir(folder)


            torch.save(result, folder+'/pipeline_result.dat')

            map_ent = pd.DataFrame(data=list(emb_training.entity_to_id.items()))
            map_ent.to_csv(folder+'/entities_to_id.tsv', sep='\t', header=False, index=False)
            map_ent = pd.DataFrame(data=list(emb_training.relation_to_id.items()))
            map_ent.to_csv(folder+'/relations_to_id.tsv', sep='\t', header=False, index=False)


            # save mappings
            result.save_to_directory(folder, save_training=True, save_metadata=True)

            # extract embeddings with gpu
            entity_embedding_tensor = result.model.entity_representations[0](indices = None)
            # save entity embeddings to a .tsv file (gpu)
            df = pd.DataFrame(data=entity_embedding_tensor.cpu().data.numpy())

            # extract embeddings with cpu
            #entity_embedding_tensor = result.model.entity_representations[0](indices=None).detach().numpy()
            # save entity embeddings to a .tsv file (cpu)
            #df = pd.DataFrame(data=entity_embedding_tensor.astype(float))

            outfile = folder + '/embeddings.tsv'
            df.to_csv(outfile, sep='\t', header=False, index=False)

            # generate .pickle of the embeddings
            gen_embedding_dict(dataset,folder)

            print(colored('Completed ' + printline,'green'))
        
        except Exception as e:

            raise e

            print(colored('An error occoured in ' + printline, 'red'))
            print(colored(e, 'red'))


if __name__ == '__main__':
    args = sys.argv[1:]
    main(args[0], args[1], args[2], args[3])

