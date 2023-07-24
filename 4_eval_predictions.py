import sys
from termcolor import colored
import shutil
import os
from pathlib import Path

def move_predictions(predictions_path):

	# clear results related to previous experiments - otherwise there will be conflicts about them
	[f.unlink() for f in Path('elliot/ssrmle_predictions/').glob("*") if f.is_file()]
	[f.unlink() for f in Path('elliot/ssrmle_results/').glob("*") if f.is_file()]

	# move the prediction file into the correct folder
	shutil.copy(predictions_path, 'elliot/ssrmle_predictions/')
	print(colored('File moved into the evaluation folder.', 'green'))

def evaluate_predictions(dataset):

	# start the evalutation
	os.system('python3 elliot/start_experiments.py '+dataset)


if __name__ == '__main__':

  # args = sys.argv[1:]
  args = list()
  args.append('drop_att')
  args.append('dbbook')
  args.append('CompGCN')
  args.append('384')
  args.append('sota_minilm-l12-v1')
  args.append('0.7')

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

  elif args[0] == 'sentence':

    # path of the output recommendation list
  	predictions_path = 'data/'+args[1]+'/predictions/sentence_'+args[4]+'.tsv'

  elif args[0] == 'concat':

    # path of the output recommendation list
  	predictions_path = 'data/'+args[1]+'/predictions/concat_'+args[2]+'_k='+args[3]+'_'+args[4]+'.tsv'

  elif args[0] == 'att':

    # path of the output recommendation list
  	predictions_path = 'data/'+args[1]+'/predictions/att_'+args[2]+'_k='+args[3]+'_'+args[4]+'.tsv'

  elif args[0] == 'drop_att':

    # path of the output recommendation list
  	predictions_path = 'data/'+args[1]+'/predictions/drop_att_'+args[2]+'_k='+args[3]+'_'+args[4]+'.tsv'

  else:

  	# error on the option
    print(colored('ERROR: '+args[0]+' is not a valid option.', 'red'))
    exit(1)

  # move the predictions, then evaluate them

  move_predictions(predictions_path)
  evaluate_predictions(args[1])			# eval the predictions by giving the dataset