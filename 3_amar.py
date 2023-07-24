import pandas as pd
import csv
import numpy as np
import json
import pickle
import os
import tensorflow as tf
import sys
from termcolor import colored
from tensorflow import keras
from numpy import loadtxt
from keras.models import Sequential
from keras.layers import Dense


# this function loads data (embeddings) to be trained/test in a unique matrix X
# whose values are then fitted by the deep model
def matching_graph_bert_ids(users, items, ratings, graph_embs, sent_embs):

  nu = []
  ni = []
  nr = []

  y_original = np.array(ratings)

  dim_embeddings = len(list(graph_embs.values())[0])

  dim_X_cols = 4
  dim_X_rows = len(users)

  X_rows = 0
  i = 0
  while i < dim_X_rows:

    user_id = users[i]
    item_id = items[i]

    check = int(user_id) in graph_embs and int(user_id) in sent_embs and int(item_id) in graph_embs and int(item_id) in sent_embs

    if check:
      X_rows += 1

    i += 1

  X = np.empty(shape=(X_rows,dim_X_cols,dim_embeddings))
  y = np.empty(shape=(X_rows))
  print(colored('Loading embeddings to be fitted/tested...','blue'))

  i=0
  c=0

  while i < dim_X_rows:

    user_id = users[i]
    item_id = items[i]

    check = int(user_id) in graph_embs and int(user_id) in sent_embs and int(item_id) in graph_embs and int(item_id) in sent_embs

    if check:

      user_graph_emb = np.array(graph_embs[int(user_id)])
      user_sent_emb = np.array(sent_embs[int(user_id)])
      item_graph_emb = np.array(graph_embs[int(item_id)])
      item_sent_emb = np.array(sent_embs[int(item_id)])

      X[c][0] = user_graph_emb
      X[c][1] = item_graph_emb
      X[c][2] = user_sent_emb
      X[c][3] = item_sent_emb

      y[c] = y_original[i]

      
      nu.append(users[i])
      ni.append(items[i])
      nr.append(ratings[i])
      
      c += 1

    i += 1

  return X[0:c], y[0:c], dim_embeddings, nu, ni, nr


def matching_graph_emb_ids(users, items, ratings, graph_emb):

  nu = []
  ni = []
  nr = []

  # esempi positivi e negativi
  y_original = np.array(ratings)
  y = np.empty(shape=(len(ratings)))

  # dimensione degli embedding
  dim_embeddings = len(list(graph_emb.values())[0])

  # numero di coppie
  dim_X_cols = 2
  dim_X_rows = len(users)

  # matrice finale di apprendimento / testing
  X = np.empty(shape=(dim_X_rows,dim_X_cols,dim_embeddings))
  print(colored('Loading embeddings to be fitted/tested...','blue'))

  # lettura degli embedding da usare per apprendere o testare
  i=0
  c=0
  while i < dim_X_rows:

      # user e item id
      user_id = users[i]
      item_id = items[i]

      if int(user_id) in graph_emb and int(item_id) in graph_emb:

        X[c][0] = np.array(graph_emb[int(user_id)])
        X[c][1] = np.array(graph_emb[int(item_id)])
        y[c] = y_original[i]
        c += 1

        nu.append(users[i])
        ni.append(items[i])
        nr.append(ratings[i])


      i += 1
      
  return X[0:c], y[0:c], dim_embeddings, nu, ni, nr


def read_ratings(filename):

  user=[]
  item=[]
  rating=[]

  with open(filename) as csv_file:

    csv_reader = csv.reader(csv_file, delimiter='\t')

    for row in csv_reader:
        user.append(int(row[0]))
        item.append(int(row[1]))
        rating.append(int(row[2]))

  return user, item, rating

def top_scores(predictions,n):

  top_n_scores = pd.DataFrame()

  for u in list(set(predictions['users'])):
    p = predictions.loc[predictions['users'] == u ]
    top_n_scores = pd.concat([top_n_scores, p.head(n)])
    #top_n_scores = top_n_scores.append(p.head(n))

  return top_n_scores

def model_basic(X, y, dim_embeddings, epochs, batch_size):
  model = keras.Sequential()

  input_users = keras.layers.Input(shape=(dim_embeddings,))

  x1 = keras.layers.Dense(512, activation=tf.nn.relu)(input_users)
  x1_2 = keras.layers.Dense(256, activation=tf.nn.relu)(x1)
  x1_3 = keras.layers.Dense(128, activation=tf.nn.relu)(x1_2)

  input_items = keras.layers.Input(shape=(dim_embeddings,))

  x2 = keras.layers.Dense(512, activation=tf.nn.relu)(input_items)
  x2_2 = keras.layers.Dense(256, activation=tf.nn.relu)(x2)
  x2_3 = keras.layers.Dense(128, activation=tf.nn.relu)(x2_2)

  concatenated = keras.layers.Concatenate()([x1_3, x2_3])

  d1 = keras.layers.Dense(64, activation=tf.nn.relu)(concatenated)
  d2 = keras.layers.Dense(64, activation=tf.nn.relu)(d1)
  out = keras.layers.Dense(1, activation=tf.nn.sigmoid)(d2)

  model = keras.models.Model(inputs=[input_users,input_items],outputs=out)
  model.compile(loss='binary_crossentropy', optimizer=tf.keras.optimizers.legacy.Adam(learning_rate=0.001, beta_1=0.9), metrics=['accuracy'])
  model.fit([X[:,0],X[:,1]], y, epochs=int(epochs), batch_size=batch_size)

  return model

def model_concatenation(X,y,dim_embeddings,epochs,batch_size):

  model = keras.Sequential()

  input_users_1 = keras.layers.Input(shape=(dim_embeddings,))
  input_items_1 = keras.layers.Input(shape=(dim_embeddings,))

  x1_user = keras.layers.Dense(256, activation=tf.nn.relu)(input_users_1)
  x1_2_user = keras.layers.Dense(128, activation=tf.nn.relu)(x1_user)
  x1_3_user = keras.layers.Dense(64, activation=tf.nn.relu)(x1_2_user)

  x1_item = keras.layers.Dense(256, activation=tf.nn.relu)(input_items_1)
  x1_2_item = keras.layers.Dense(128, activation=tf.nn.relu)(x1_item)
  x1_3_item = keras.layers.Dense(64, activation=tf.nn.relu)(x1_2_item)

  input_users_2 = keras.layers.Input(shape=(dim_embeddings,))
  input_items_2 = keras.layers.Input(shape=(dim_embeddings,))

  x2_user = keras.layers.Dense(256, activation=tf.nn.relu)(input_users_2)
  x2_2_user = keras.layers.Dense(128, activation=tf.nn.relu)(x2_user)
  x2_3_user = keras.layers.Dense(64, activation=tf.nn.relu)(x2_2_user)

  x2_item = keras.layers.Dense(256, activation=tf.nn.relu)(input_items_2)
  x2_2_item = keras.layers.Dense(128, activation=tf.nn.relu)(x2_item)
  x2_3_item = keras.layers.Dense(64, activation=tf.nn.relu)(x2_2_item)
  
  concatenated_1 = keras.layers.Concatenate()([x1_3_user, x2_3_user])
  dense_user = keras.layers.Dense(64, activation=tf.nn.relu)(concatenated_1)
  dense_user_2 = keras.layers.Dense(32, activation=tf.nn.relu)(dense_user)
  
  concatenated_2 = keras.layers.Concatenate()([x1_3_item, x2_3_item])
  dense_item = keras.layers.Dense(64, activation=tf.nn.relu)(concatenated_2)
  dense_item_2 = keras.layers.Dense(32, activation=tf.nn.relu)(dense_item)

  concatenated = keras.layers.Concatenate()([dense_user_2, dense_item_2])
  dense = keras.layers.Dense(32, activation=tf.nn.relu)(concatenated)
  dense2 = keras.layers.Dense(16, activation=tf.nn.relu)(dense)
  dense3 = keras.layers.Dense(8, activation=tf.nn.relu)(dense2)
  out = keras.layers.Dense(1, activation=tf.nn.sigmoid)(dense3)

  model = keras.models.Model(inputs=[input_users_1,input_items_1,input_users_2,input_items_2],outputs=out)
  model.compile(loss='binary_crossentropy', optimizer=tf.keras.optimizers.legacy.Adam(learning_rate=0.001, beta_1=0.9), metrics=['accuracy'])
  model.fit([X[:,0],X[:,1],X[:,2],X[:,3]], y, epochs=int(epochs), batch_size=batch_size)
  
  return model

def model_selfatt_crossatt(X,y,dim_embeddings,epochs,batch_size):

  model = keras.Sequential()

  input_users_1 = keras.layers.Input(shape=(dim_embeddings,))
  input_items_1 = keras.layers.Input(shape=(dim_embeddings,))

  x1_user = keras.layers.Dense(512, activation=tf.nn.relu)(input_users_1)
  x1_2_user = keras.layers.Dense(256, activation=tf.nn.relu)(x1_user)
  x1_3_user = keras.layers.Dense(128, activation=tf.nn.relu)(x1_2_user)

  x1_item = keras.layers.Dense(512, activation=tf.nn.relu)(input_items_1)
  x1_2_item = keras.layers.Dense(256, activation=tf.nn.relu)(x1_item)
  x1_3_item = keras.layers.Dense(128, activation=tf.nn.relu)(x1_2_item)

  input_users_2 = keras.layers.Input(shape=(dim_embeddings,))
  input_items_2 = keras.layers.Input(shape=(dim_embeddings,))

  x2_user = keras.layers.Dense(512, activation=tf.nn.relu)(input_users_2)
  x2_2_user = keras.layers.Dense(256, activation=tf.nn.relu)(x2_user)
  x2_3_user = keras.layers.Dense(128, activation=tf.nn.relu)(x2_2_user)

  x2_item = keras.layers.Dense(512, activation=tf.nn.relu)(input_items_2)
  x2_2_item = keras.layers.Dense(256, activation=tf.nn.relu)(x2_item)
  x2_3_item = keras.layers.Dense(128, activation=tf.nn.relu)(x2_2_item)

  # self attenzione 1 - merge graph user e word user
  concat_user = keras.layers.Concatenate()([x1_3_user, x2_3_user])
  attention_w_user = keras.layers.Dense(128, activation='softmax')(concat_user)
  merged_user = attention_w_user * x1_3_user + (1 - attention_w_user) * x2_3_user

  # self attenzione 2 - merge graph item e word item
  concat_item = keras.layers.Concatenate()([x1_3_item, x2_3_item])
  attention_w_item = keras.layers.Dense(128, activation='softmax')(concat_item)
  merged_item = attention_w_item * x1_3_item + (1 - attention_w_item) * x2_3_item

  # cross attenzione - merge dei due merged
  attention_weights = keras.layers.Dot(axes=-1)([merged_user, merged_item])
  attention_weights = keras.layers.Dense(128, activation='softmax')(attention_weights)
  merged = keras.layers.Add()([merged_user * attention_weights, merged_item * (1 - attention_weights)])

  merged2 = keras.layers.Dense(64, activation=tf.nn.relu)(merged)
  merged3 = keras.layers.Dense(32, activation=tf.nn.relu)(merged2)
  merged4 = keras.layers.Dense(16, activation=tf.nn.relu)(merged3)
  merged5 = keras.layers.Dense(8, activation=tf.nn.relu)(merged4)
  out = keras.layers.Dense(1, activation=tf.nn.sigmoid)(merged5)

  model = keras.models.Model(inputs=[input_users_1,input_items_1,input_users_2,input_items_2],outputs=out)
  model.compile(loss='binary_crossentropy', optimizer=tf.keras.optimizers.legacy.Adam(learning_rate=0.001, beta_1=0.9), metrics=['accuracy'])
  model.fit([X[:,0],X[:,1],X[:,2],X[:,3]], y, epochs=int(epochs), batch_size=batch_size)
  
  return model

def model_dropout_selfatt_crossatt(X,y,dim_embeddings,epochs,batch_size, value):

  model = keras.Sequential()

  input_users_1 = keras.layers.Input(shape=(dim_embeddings,))
  input_items_1 = keras.layers.Input(shape=(dim_embeddings,))

  x1_user_drop = keras.layers.Dropout(value, input_shape=(dim_embeddings,))(input_users_1)
  x1_item_drop = keras.layers.Dropout(value, input_shape=(dim_embeddings,))(input_items_1)

  x1_user = keras.layers.Dense(512, activation=tf.nn.relu)(x1_user_drop)
  x1_2_user = keras.layers.Dense(256, activation=tf.nn.relu)(x1_user)
  x1_3_user = keras.layers.Dense(128, activation=tf.nn.relu)(x1_2_user)

  x1_item = keras.layers.Dense(512, activation=tf.nn.relu)(x1_item_drop)
  x1_2_item = keras.layers.Dense(256, activation=tf.nn.relu)(x1_item)
  x1_3_item = keras.layers.Dense(128, activation=tf.nn.relu)(x1_2_item)

  input_users_2 = keras.layers.Input(shape=(dim_embeddings,))
  input_items_2 = keras.layers.Input(shape=(dim_embeddings,))

  x2_user_drop = keras.layers.Dropout(value, input_shape=(dim_embeddings,))(input_users_2)
  x2_item_drop = keras.layers.Dropout(value, input_shape=(dim_embeddings,))(input_items_2)

  x2_user = keras.layers.Dense(512, activation=tf.nn.relu)(x2_user_drop)
  x2_2_user = keras.layers.Dense(256, activation=tf.nn.relu)(x2_user)
  x2_3_user = keras.layers.Dense(128, activation=tf.nn.relu)(x2_2_user)

  x2_item = keras.layers.Dense(512, activation=tf.nn.relu)(x2_item_drop)
  x2_2_item = keras.layers.Dense(256, activation=tf.nn.relu)(x2_item)
  x2_3_item = keras.layers.Dense(128, activation=tf.nn.relu)(x2_2_item)

  # self attenzione 1 - merge graph user e word user
  concat_user = keras.layers.Concatenate()([x1_3_user, x2_3_user])
  attention_w_user = keras.layers.Dense(128, activation='softmax')(concat_user)
  merged_user = attention_w_user * x1_3_user + (1 - attention_w_user) * x2_3_user

  # self attenzione 2 - merge graph item e word item
  concat_item = keras.layers.Concatenate()([x1_3_item, x2_3_item])
  attention_w_item = keras.layers.Dense(128, activation='softmax')(concat_item)
  merged_item = attention_w_item * x1_3_item + (1 - attention_w_item) * x2_3_item

  # cross attenzione - merge dei due merged
  attention_weights = keras.layers.Dot(axes=-1)([merged_user, merged_item])
  attention_weights = keras.layers.Dense(128, activation='softmax')(attention_weights)
  merged = keras.layers.Add()([merged_user * attention_weights, merged_item * (1 - attention_weights)])

  merged2 = keras.layers.Dense(64, activation=tf.nn.relu)(merged)
  merged3 = keras.layers.Dense(32, activation=tf.nn.relu)(merged2)
  merged4 = keras.layers.Dense(16, activation=tf.nn.relu)(merged3)
  merged5 = keras.layers.Dense(8, activation=tf.nn.relu)(merged4)
  out = keras.layers.Dense(1, activation=tf.nn.sigmoid)(merged5)

  model = keras.models.Model(inputs=[input_users_1,input_items_1,input_users_2,input_items_2],outputs=out)
  model.compile(loss='binary_crossentropy', optimizer=tf.keras.optimizers.legacy.Adam(learning_rate=0.001, beta_1=0.9), metrics=['accuracy'])
  model.fit([X[:,0],X[:,1],X[:,2],X[:,3]], y, epochs=int(epochs), batch_size=batch_size)
  
  return model


def train_graph(dataset, graph_emb_model, graph_emb_dim, epochs):

  # paths of the embedding
  source_graph_path = 'data/'+dataset+'/'+dataset+'_'+graph_emb_model+'_k='+graph_emb_dim+'.pickle'

  if not os.path.isfile(source_graph_path):
    print(colored('ERROR: Graph embeddings not existing', 'red'))
    exit(1)
  
  # path of the recsys output model
  rec_model_path = 'data/'+dataset+'/recmodels/graph_'+graph_emb_model+'.keras'

  # path of the output recommendation list
  predictions_path = 'data/'+dataset+'/predictions/graph_'+graph_emb_model+'_k='+graph_emb_dim+'.tsv'
  predictions_path_short = 'data/'+dataset+'/predictions.tsv'
  
  # read training data
  users, items, ratings = read_ratings('data/dbbook/train.tsv')

  # read graph embeddings
  graph_emb = pickle.load(open(source_graph_path, 'rb'))

  # il the model already exixts, it's loaded
  if os.path.exists(rec_model_path) and False:

    recsys_model = tf.keras.models.load_model(rec_model_path)
    print(colored('Model loaded', 'blue'))

  # otherwise it's trained
  else:

    print(colored('Matching data for training...','blue'), end=' ')
    print(colored('Matched.', 'green'))
    X, y, dim_embeddings, _, _, _ = matching_graph_emb_ids(users, items, ratings, graph_emb)
    
    # training the model
    recsys_model = model_basic(X,y,dim_embeddings,epochs=epochs,batch_size=512)

    # saving the model
    recsys_model.save(rec_model_path)

  # read test ratings to be predicted
  users, items, ratings = read_ratings('data/dbbook/test.tsv')

  # embeddings for test
  X, y, dim_embeddings, nu, ni, nr = matching_graph_emb_ids(users, items, ratings, graph_emb)

  # predict   
  print(colored('\tPredicting...', 'blue'))
  score = recsys_model.predict([X[:,0], X[:,1]])

  # write predictions
  print(colored('\tComputing predictions...', 'blue'))
  score = score.reshape(1, -1)[0,:]
  predictions = pd.DataFrame()
  predictions['users'] = np.array(nu)
  predictions['items'] = np.array(ni)
  predictions['scores'] = score

  predictions = predictions.sort_values(by=['users', 'scores'], ascending=[True, False])

  # write top 5 predictions
  top_5_scores = top_scores(predictions,5)
  top_5_scores.to_csv(predictions_path, sep='\t', header=False, index=False)
  top_5_scores.to_csv(predictions_path_short, sep='\t', header=False, index=False)
  print(colored('\tTop 5 predicted', 'green'))




def train_sentence(dataset, sent_emb_model, epochs):

  # paths of the embedding
  source_sent_path = 'data/'+dataset+'/'+dataset+'_'+sent_emb_model+'.pickle'

  if not os.path.isfile(source_sent_path):
    print(colored('ERROR: Sentence embeddings not existing', 'red'))
    exit(1)
  
  # path of the recsys output model
  rec_model_path = 'data/'+dataset+'/recmodels/sentence_'+sent_emb_model+'.keras'

  # path of the output recommendation list
  predictions_path = 'data/'+dataset+'/predictions/sentence_'+sent_emb_model+'.tsv'
  predictions_path_short = 'data/'+dataset+'/predictions.tsv'

  # read training data
  users, items, ratings = read_ratings('data/dbbook/train.tsv')

  # read word embeddings
  sent_emb = pickle.load(open(source_sent_path, 'rb'))


  # il the model already exixts, it's loaded
  if os.path.exists(rec_model_path) and False:

    recsys_model = tf.keras.models.load_model(rec_model_path)
    print(colored('Model loaded', 'blue'))

  # otherwise it's trained
  else:

    print(colored('Matching data for training...','blue'), end=' ')
    print(colored('Matched.', 'green'))
    X, y, dim_embeddings, _, _, _ = matching_graph_emb_ids(users, items, ratings, sent_emb)
    
    # training the model
    recsys_model = model_basic(X,y,dim_embeddings,epochs=epochs,batch_size=512)

    # saving the model
    recsys_model.save(rec_model_path)

  # read test ratings to be predicted
  users, items, ratings = read_ratings('data/dbbook/test.tsv')

  # embeddings for test
  X, y, dim_embeddings, nu, ni, nr = matching_graph_emb_ids(users, items, ratings, sent_emb)

  # predict   
  print(colored('\tPredicting...', 'blue'))
  score = recsys_model.predict([X[:,0], X[:,1]])

  # write predictions
  print(colored('\tComputing predictions...', 'blue'))
  score = score.reshape(1, -1)[0,:]
  predictions = pd.DataFrame()
  predictions['users'] = np.array(nu)
  predictions['items'] = np.array(ni)
  predictions['scores'] = score

  predictions = predictions.sort_values(by=['users', 'scores'], ascending=[True, False])

  # write top 5 predictions
  top_5_scores = top_scores(predictions,5)
  top_5_scores.to_csv(predictions_path, sep='\t', header=False, index=False)
  top_5_scores.to_csv(predictions_path_short, sep='\t', header=False, index=False)
  print(colored('\tTop 5 predicted', 'green'))



def train_concat(dataset, graph_emb_model, graph_emb_dim, sent_emb_model, epochs):

  # paths of the embedding
  source_graph_path = 'data/'+dataset+'/'+dataset+'_'+graph_emb_model+'_k='+graph_emb_dim+'.pickle'
  source_sent_path = 'data/'+dataset+'/'+dataset+'_'+sent_emb_model+'.pickle'

  if not os.path.isfile(source_graph_path):
    print(colored('ERROR: Graph embeddings not existing', 'red'))
    exit(1)

  if not os.path.isfile(source_sent_path):
    print(colored('ERROR: Sentence embeddings not existing', 'red'))
    exit(1)
  
  # path of the recsys output model
  rec_model_path = 'data/'+dataset+'/recmodels/concat_'+graph_emb_model+'_k='+graph_emb_dim+'_'+sent_emb_model+'.keras'

  # path of the output recommendation list
  predictions_path = 'data/'+dataset+'/predictions/concat_'+graph_emb_model+'_k='+graph_emb_dim+'_'+sent_emb_model+'.tsv'
  predictions_path_short = 'data/'+dataset+'/predictions.tsv'

  # read training data
  users, items, ratings = read_ratings('data/dbbook/train.tsv')

  # read graph and word embedding
  graph_emb = pickle.load(open(source_graph_path, 'rb'))
  sent_emb = pickle.load(open(source_sent_path, 'rb'))


  # il the model already exixts, it's loaded
  if os.path.exists(rec_model_path) and False:

    recsys_model = tf.keras.models.load_model(rec_model_path)
    print(colored('Model loaded', 'blue'))

  # otherwise it's trained
  else:

    print(colored('Matching data for training...','blue'), end=' ')
    print(colored('Matched.', 'green'))
    X, y, dim_embeddings, _, _, _ = matching_graph_bert_ids(users, items, ratings, graph_emb, sent_emb)
    
    # training the model
    recsys_model = model_concatenation(X,y,dim_embeddings,epochs=epochs,batch_size=512)

    # saving the model
    recsys_model.save(rec_model_path)

  # read test ratings to be predicted
  users, items, ratings = read_ratings('data/dbbook/test.tsv')

  # embeddings for test
  X, y, dim_embeddings, nu, ni, nr = matching_graph_bert_ids(users, items, ratings, graph_emb, sent_emb)

  # predict   
  print(colored('\tPredicting...', 'blue'))
  score = recsys_model.predict([X[:,0], X[:,1], X[:,2], X[:,3]])

  # write predictions
  print(colored('\tComputing predictions...', 'blue'))
  score = score.reshape(1, -1)[0,:]
  predictions = pd.DataFrame()
  predictions['users'] = np.array(nu)
  predictions['items'] = np.array(ni)
  predictions['scores'] = score

  predictions = predictions.sort_values(by=['users', 'scores'], ascending=[True, False])

  # write top 5 predictions
  top_5_scores = top_scores(predictions,5)
  top_5_scores.to_csv(predictions_path, sep='\t', header=False, index=False)
  top_5_scores.to_csv(predictions_path_short, sep='\t', header=False, index=False)
  print(colored('\tTop 5 predicted', 'green'))



def train_att(dataset, graph_emb_model, graph_emb_dim, sent_emb_model, epochs):

  # paths of the embedding
  source_graph_path = 'data/'+dataset+'/'+dataset+'_'+graph_emb_model+'_k='+graph_emb_dim+'.pickle'
  source_sent_path = 'data/'+dataset+'/'+dataset+'_'+sent_emb_model+'.pickle'

  if not os.path.isfile(source_graph_path):
    print(colored('ERROR: Graph embeddings not existing', 'red'))
    exit(1)

  if not os.path.isfile(source_sent_path):
    print(colored('ERROR: Sentence embeddings not existing', 'red'))
    exit(1)
  
  # path of the recsys output model
  rec_model_path = 'data/'+dataset+'/recmodels/att_'+graph_emb_model+'_k='+graph_emb_dim+'_'+sent_emb_model+'.keras'

  # path of the output recommendation list
  predictions_path = 'data/'+dataset+'/predictions/att_'+graph_emb_model+'_k='+graph_emb_dim+'_'+sent_emb_model+'.tsv'
  predictions_path_short = 'data/'+dataset+'/predictions.tsv'

  # read training data
  users, items, ratings = read_ratings('data/dbbook/train.tsv')

  # read graph and word embedding
  graph_emb = pickle.load(open(source_graph_path, 'rb'))
  sent_emb = pickle.load(open(source_sent_path, 'rb'))


  # il the model already exixts, it's loaded
  if os.path.exists(rec_model_path) and False:

    recsys_model = tf.keras.models.load_model(rec_model_path)
    print(colored('Model loaded', 'blue'))

  # otherwise it's trained
  else:

    print(colored('Matching data for training...','blue'), end=' ')
    print(colored('Matched.', 'green'))
    X, y, dim_embeddings, _, _, _ = matching_graph_bert_ids(users, items, ratings, graph_emb, sent_emb)
    
    # training the model
    recsys_model = model_selfatt_crossatt(X,y,dim_embeddings,epochs=epochs,batch_size=512)

    # saving the model
    recsys_model.save(rec_model_path)

  # read test ratings to be predicted
  users, items, ratings = read_ratings('data/dbbook/test.tsv')

  # embeddings for test
  X, y, dim_embeddings, nu, ni, nr = matching_graph_bert_ids(users, items, ratings, graph_emb, sent_emb)

  # predict   
  print(colored('\tPredicting...', 'blue'))
  score = recsys_model.predict([X[:,0], X[:,1], X[:,2], X[:,3]])

  # write predictions
  print(colored('\tComputing predictions...', 'blue'))
  score = score.reshape(1, -1)[0,:]
  predictions = pd.DataFrame()
  predictions['users'] = np.array(nu)
  predictions['items'] = np.array(ni)
  predictions['scores'] = score

  predictions = predictions.sort_values(by=['users', 'scores'], ascending=[True, False])

  # write top 5 predictions
  top_5_scores = top_scores(predictions,5)
  top_5_scores.to_csv(predictions_path, sep='\t', header=False, index=False)
  top_5_scores.to_csv(predictions_path_short, sep='\t', header=False, index=False)
  print(colored('\tTop 5 predicted', 'green'))



def train_drop_att(dataset, graph_emb_model, graph_emb_dim, sent_emb_model, dropout_rate, epochs):

  # paths of the embedding
  source_graph_path = 'data/'+dataset+'/'+dataset+'_'+graph_emb_model+'_k='+graph_emb_dim+'.pickle'
  source_sent_path = 'data/'+dataset+'/'+dataset+'_'+sent_emb_model+'.pickle'

  if not os.path.isfile(source_graph_path):
    print(colored('ERROR: Graph embeddings not existing', 'red'))
    print(source_graph_path)
    exit(1)

  if not os.path.isfile(source_sent_path):
    print(colored('ERROR: Sentence embeddings not existing', 'red'))
    exit(1)
  
  # path of the recsys output model
  rec_model_path = 'data/'+dataset+'/recmodels/drop_'+str(dropout_rate.replace('.',''))+'_att_'+graph_emb_model+'_k='+graph_emb_dim+'_'+sent_emb_model+'.keras'

  # path of the output recommendation list
  predictions_path = 'data/'+dataset+'/predictions/drop_att_'+graph_emb_model+'_k='+graph_emb_dim+'_'+sent_emb_model+'.tsv'
  predictions_path_short = 'data/'+dataset+'/predictions.tsv'

  # read training data
  users, items, ratings = read_ratings('data/dbbook/train.tsv')

  # read graph and word embedding
  graph_emb = pickle.load(open(source_graph_path, 'rb'))
  sent_emb = pickle.load(open(source_sent_path, 'rb'))


  # il the model already exixts, it's loaded
  if os.path.exists(rec_model_path) and False:

    recsys_model = tf.keras.models.load_model(rec_model_path)
    print(colored('Model loaded', 'blue'))

  # otherwise it's trained
  else:

    print(colored('Matching data for training...','blue'), end=' ')
    print(colored('Matched.', 'green'))
    X, y, dim_embeddings, _, _, _ = matching_graph_bert_ids(users, items, ratings, graph_emb, sent_emb)
    
    # training the model
    recsys_model = model_dropout_selfatt_crossatt(X,y,dim_embeddings,epochs=epochs,batch_size=512, value=float(dropout_rate))

    # saving the model
    recsys_model.save(rec_model_path)

  # read test ratings to be predicted
  users, items, ratings = read_ratings('data/dbbook/test.tsv')

  # embeddings for test
  X, y, dim_embeddings, nu, ni, nr = matching_graph_bert_ids(users, items, ratings, graph_emb, sent_emb)

  # predict   
  print(colored('\tPredicting...', 'blue'))
  score = recsys_model.predict([X[:,0], X[:,1], X[:,2], X[:,3]])

  # write predictions
  print(colored('\tComputing predictions...', 'blue'))
  score = score.reshape(1, -1)[0,:]
  predictions = pd.DataFrame()
  predictions['users'] = np.array(nu)
  predictions['items'] = np.array(ni)
  predictions['scores'] = score

  predictions = predictions.sort_values(by=['users', 'scores'], ascending=[True, False])

  # write top 5 predictions
  top_5_scores = top_scores(predictions,5)
  top_5_scores.to_csv(predictions_path, sep='\t', header=False, index=False)
  top_5_scores.to_csv(predictions_path_short, sep='\t', header=False, index=False)
  print(colored('\tTop 5 predicted', 'green'))


if __name__ == '__main__':

  args = sys.argv[1:]

  # List of the parameters
  # args[0] --> type of the model: 'graph', 'sentence', 'concat', 'att', 'drop_att'
  # args[1] --> dataset
  # args[2] --> graph encoder
  # args[3] --> size of the graph encoder
  # args[4] --> sentence encoder
  # args[5] --> dropout value
  # args[6] --> epochs

  if args[0] == 'graph':

    # this considers only graph encoder (datset, graph encoder, dimension, epochs)
    train_graph(args[1], args[2], args[3], args[6])

  elif args[0] == 'sentence':

    # only sentence encoder (dataset, sentence encoder)
    train_sentence(args[1], args[4], args[6]) 

  elif args[0] == 'concat':

    # concatenate graph and sentence embeddings (datset, graph encoder, dimension, sentence encoder, epochs)
    train_concat(args[1], args[2], args[3], args[4], args[6])

  elif args[0] == 'att':

    # concatenate graph and sentence embeddings (datset, graph encoder, dimension, sentence encoder, epochs)
    train_att(args[1], args[2], args[3], args[4], args[6])

  elif args[0] == 'drop_att':

    # concatenate graph and sentence embeddings (datset, graph encoder, dimension, sentence encoder, dropout rate, epochs)
    train_drop_att(args[1], args[2], args[3], args[4], args[5], args[6])

  else:

    print(colored('ERROR: '+args[0]+' is not a valid option.', 'red'))
    exit(1)











