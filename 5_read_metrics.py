import mlflow
import pandas as pd
import os
import sys
from termcolor import colored


if __name__ == '__main__':

  args = sys.argv[1:]

  # List of the parameters
  # args[0] --> dataset
  # args[1] --> graph encoder
  # args[2] --> size of the graph encoder
  # args[3] --> sentence encoder
  # args[4] --> type of the model: 'graph', 'sentence', 'concat', 'att', 'drop_att'
  # args[5] --> dropout value
  # args[5] --> epochs
  
  if args[0] == 'graph':

    # path of the output recommendation list
    predictions_path = 'data/'+args[1]+'/predictions/graph_'+args[2]+'_k='+args[3]+'.tsv'
    param_dict = {'config': 'graph',
                  'dataset': args[1],
                  'graph_enc': args[2], 
                  'k': int(args[3]),
                  'epochs': int(args[6])}

  elif args[0] == 'sentence':

    # path of the output recommendation list
    predictions_path = 'data/'+args[1]+'/predictions/sentence_'+args[2]+'.tsv'
    param_dict = {'config': 'sentence',
                  'dataset': args[1],
                  'sentence_enc': args[2],
                  'epochs': int(args[6])}

  elif args[0] == 'concat':

    # path of the output recommendation list
  	predictions_path = 'data/'+args[1]+'/predictions/concat_'+args[2]+'_k='+args[3]+'_'+args[4]+'.tsv'
  	param_dict = {'config': 'concat',
                  'dataset': args[1],
                  'graph_enc': args[2],
                  'k': int(args[3]),
                  'sentence_enc': args[4],
                  'epochs': int(args[6])}

  elif args[0] == 'att':

    # path of the output recommendation list
    predictions_path = 'data/'+args[1]+'/predictions/att_'+args[2]+'_k='+args[3]+'_'+args[4]+'.tsv'
    param_dict = {'config': 'att',
                  'dataset': args[1],
                  'graph_enc': args[2],
                  'k': int(args[3]),
                  'sentence_enc': args[4],
                  'epochs': int(args[6])}

  elif args[0] == 'drop_att':

    # path of the output recommendation list
    predictions_path = 'data/'+args[1]+'/predictions/drop_att_'+str(args[5].replace('.',''))+args[2]+'_k='+args[3]+'_'+args[4]+'.tsv'
    param_dict = {'config': 'drop_att',
                   'dataset': args[1],
                   'graph_enc': args[2],
                   'k': int(args[3]),
                   'sentence_enc': args[4],
                   'drop_value': float(args[5]),
                   'epochs': int(args[6])}


  else:

  	# error on the option
    print(colored('ERROR: '+args[0]+' is not a valid option.', 'red'))
    exit(1)

  # set a meaningful project name
  project_name = predictions_path.replace(".tsv","").split('/')[-1]

  # start loggin experiments and save current parameters
  # mlflow.start_run(run_name = project_name)
  mlflow.start_run()
  mlflow.log_params(param_dict)

  # read the metrics computed by elliot from a .tsv file
  df = pd.read_csv('elliot/ssrmle_results/metrics.tsv', sep='\t')
  dict_metrics = df.to_dict('records')[0]
  dict_metrics.pop('model', None)

  print(dict_metrics)
  
  # log metric values
  mlflow.log_metrics(dict_metrics) 

  # stop experiment
  mlflow.end_run()
  mlflow.end_run()