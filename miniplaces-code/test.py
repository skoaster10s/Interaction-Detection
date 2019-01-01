import gc
import sys
from torch.autograd import Variable
import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler
torch.backends.cudnn.benchmark=True

import dataset
from models.AlexNet import *
from models.ResNet import *


def test(model_name):
	# Parameters
	output_period = 100
	batch_size = 100
	maxk = 5

	print("Using: " + model_name)

	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
	model = resnet_18()
	model.load_state_dict(torch.load('models/'+model_name))
	model = model.to(device)

	_, test_loader = dataset.get_val_test_loaders(batch_size)
	num_test_batches = len(test_loader)
	filenames = get_test_filenames(test_loader)
	full_predictions = []
	num_test_batches = len(test_loader)

	for batch_num, (inputs, labels) in enumerate(test_loader, 1):
		if batch_num % output_period == 0:
			print('Progress: %.2f' % (batch_num*1.0/num_test_batches))
		inputs = inputs.to(device)
		labels = labels.to(device)
		outputs = model(inputs)
		pred = outputs.topk(maxk, 1)[1]

		corresponding_files = filenames[(batch_num-1)*batch_size:batch_num*batch_size]
		full_predictions += list(zip(corresponding_files, pred))
	
	write_full_predictions_to_file(full_predictions)

def get_test_filenames(loader):
	return list(map(lambda x: 'test/' + x[0].split('/')[-1], loader.dataset.samples))

def write_full_predictions_to_file(full_predictions, file='predictions.txt'):
	output = ''
	for f, preds in full_predictions:
		output += f + ' ' + ' '.join([str(int(p)) for p in preds]) + '\n'
	
	with open(file, 'w') as f:
		f.write(output[:-1])


if __name__ == "__main__":
	model_name = 'model.10'
	if len(sys.argv) == 2:
		model_name = sys.argv[1]

	print('Starting testing')
	test(model_name)
	print('Testing terminated')

