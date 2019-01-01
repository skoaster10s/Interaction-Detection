import gc
import sys
from torch.autograd import Variable
import torch
import torch.nn as nn
import torch.optim as optim
torch.backends.cudnn.benchmark=True

import dataset
from models.AlexNet import *
from models.ResNet import *

def run():
	# Parameters
	num_epochs = 40
	output_period = 100
	batch_size = 100

	# setup the device for running
	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
	model = resnet_18()
	model = model.to(device)

	train_loader, val_loader = dataset.get_data_loaders(batch_size)
	num_train_batches = len(train_loader)

	criterion = nn.CrossEntropyLoss().to(device)
	# TODO: optimizer is currently unoptimized
	# there's a lot of room for improvement/different optimizers
	optimizer = optim.SGD(model.parameters(), lr=0.55, momentum=0.4, weight_decay=1e-4)

	epoch = 1
	while epoch <= num_epochs:
		running_loss = 0.0
		for param_group in optimizer.param_groups:
			print('Current learning rate: ' + str(param_group['lr']))
		model.train()

		for batch_num, (inputs, labels) in enumerate(train_loader, 1):
			inputs = inputs.to(device)
			labels = labels.to(device)

			optimizer.zero_grad()
			outputs = model(inputs)
			loss = criterion(outputs, labels)
			loss.backward()

			optimizer.step()
			running_loss += loss.item()

			if batch_num % output_period == 0:
				print('[%d:%.2f] loss: %.3f' % (
					epoch, batch_num*1.0/num_train_batches,
					running_loss/output_period
					))
				running_loss = 0.0
				gc.collect()

		gc.collect()
		# save after every epoch
		torch.save(model.state_dict(), "models/model.%d" % epoch)

		# TODO: Calculate classification error and Top-5 Error
		# on training and validation datasets here
		train_error = compute_error_rates(model, train_loader, device)
		valid_error = compute_error_rates(model, val_loader, device)

		print('Train:\tTop-1 Error: %.3f\tTop-5 Error: %.3f' % train_error)
		print('Valid:\tTop-1 Error: %.3f\tTop-5 Error: %.3f' % valid_error)

		gc.collect()
		epoch += 1

def compute_error_rates(model, loader, device):
	model.eval()
	maxk = 5
	top_1_correct = 0.
	top_5_correct = 0.
	total = 0.

	for batch_num, (inputs, labels) in enumerate(loader, 1):
		inputs = inputs.to(device)
		labels = labels.to(device)
		outputs = model(inputs)

		pred = outputs.topk(maxk, 1)[1].t()
		correct = pred.eq(labels.view(1, -1).expand_as(pred))

		top_1_correct += correct[:1].view(-1).float().sum(0, keepdim=True)
		top_5_correct += correct[:5].view(-1).float().sum(0, keepdim=True)
		total += inputs.size(0)

	return ((1-top_1_correct/total),(1-top_5_correct/total))


print('Starting training')
run()
print('Training terminated')
