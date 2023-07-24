import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from termcolor import colored
import os
import sys
import pickle


def main(dataset, model_name):

  # descriptive file in the format 'item;description'
  df = pd.read_csv("data/"+dataset+"/sentence_dummy/dummy_text.csv", sep=';;')
  print(colored('Item descriptions read.','green'))

  # get ids and descriptions of the items
  ids = np.array(df["item"])
  descriptions= np.array(df["description"])

  # word model chosen
  model = SentenceTransformer('sentence-transformers/'+model_name)
  model = SentenceTransformer('sentence-transformers/all-MiniLM-L12-v2')

  embeddings = []

  print(colored('Starting learning with '+model_name, 'blue'))

  # encoding sentences
  i = 0
  for cont in descriptions:

    i += 1

    # learn the embedding and add it to the set of embeddings
    try:
      embedding = model.encode(cont)
      embeddings.append(embedding)
      print(colored('Learned '+str(i), 'green'))
    except:
      embedding = np.zeros(len(embedding))
      embeddings.append(embedding)

  # create dictionary
  dictionary = {}

  i = 0
  while i < len(ids):
    dictionary[ids[i]] = embeddings[i]
    i += 1

  # size of the matrix of embeddings learnt - just to verify the output
  print(str(len(embeddings))+'x'+str(len(embeddings[0])))
  
  print(colored('Sentence embeddings learnt. Starting building user profiles...', 'green'))

  # read interactions file
  interactions = open("data/"+dataset+"/sentence_dummy/dummy_interactions.tsv", 'r', encoding='utf-8')

  usr_likes = dict()

  for line in interactions:
    user, item, rating = int(line.strip().split('\t')[0]), int(line.strip().split('\t')[1]), int(line.strip().split('\t')[2])
    
    # add user to the list of profiles to be built
    if user not in usr_likes:
      usr_likes[user] = set()

    # add the liked item id
    if rating == 1:
      usr_likes[user].add(item)

  # compute the user profile as the centroid of the liked books (embeddings)
  for user in usr_likes:

    # get the embeddings of the liked books
    items = []
    for item in usr_likes[user]:
      item_emb = dictionary[item]
      items.append(item_emb)

    # compute the centroid and add it to the dictonary
    user_emb = np.mean(items, axis=0)
    dictionary[user] = user_emb

  # list of entities for which we computed the embeddings - it is composed of users and items
  print(dictionary.keys())

  # save the embeddings
  # dummy name
  pickle.dump(dictionary,open("data/"+dataset+"/sentence_dummy/dummy_embeddings_"+model_name+".pickle", 'wb')) 

  print(colored('User profiles learnt. Embedding saved.','green'))
  # real name
  #pickle.dump(dictionary,open("data/"+dataset+"_"+model_name+".pickle", 'wb')) # real name

if __name__ == '__main__':

  args = sys.argv[1:]

  # check the available model
  if args[1] == 'all':
    model = 'all-MiniLM-L12-v2'
  elif args[1] == 'para':
    model = 'paraphrase-MiniLM-L6-v2'
  else:
    print(colored('Sentence model not available.','red'))
    exit(0)


  main(args[0], model)



