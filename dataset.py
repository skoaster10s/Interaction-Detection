import os
import glob
import gc
import sys
from skimage import io, transform
from torch.autograd import Variable
import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler
from torch.utils.data import Dataset, DataLoader
torch.backends.cudnn.benchmark=True

from torchvision import datasets, transforms
import torchvision.models as models


data_root = './data/'
train_root = data_root + 'train'
val_root = data_root + 'val'
test_root = data_root + 'test'

batch_size = 2

class MultiResNet(nn.Module):
	def __init__(self):
		super(MultiResNet, self).__init__()
		self.resnet1 = models.resnet18(pretrained=True)
		self.resnet2 = models.resnet18(pretrained=True)
		self.resnet3 = models.resnet18(pretrained=True)
		self.fc1 = nn.Linear(3000, 100)
		self.fc2 = nn.Linear(100, 1)

	def forward(self, x1, x2, x3):
		x1 = self.resnet1(x1)
		x2 = self.resnet2(x2)
		x3 = self.resnet3(x3)

		x1 = x1.view(x1.size(0), -1)
		x2 = x2.view(x2.size(0), -1)
		x3 = x3.view(x3.size(0), -1)

		x = torch.cat((x1, x2, x3), dim=1)
		x = self.fc1(x)
		x = self.fc2(x)
		return x

class InteractionsDataset(Dataset):
	"""Interactions Dataset"""

	def __init__(self, root_dir, transform=None):
		"""
		Args:
			csv_file (string): Path to the csv file with annotations.
			root_dir (string): Directory with all the images.
			transform (callable, optional): Optional transform to be applied
				on a sample.
		"""
		self.root_dir = root_dir
		positive_imgs = sorted(glob.glob(root_dir + '/1/*.png'))
		positive_imgs = [positive_imgs[3*i:3*i+3] + [1] for i in range(len(positive_imgs)//3)]
		negative_imgs = sorted(glob.glob(root_dir + '/0/*.png'))
		negative_imgs = [negative_imgs[3*i:3*i+3] + [0] for i in range(len(negative_imgs)//3)]
		self.imgs = positive_imgs + negative_imgs
		self.transform = transform

	def __len__(self):
		return len(self.imgs)

	def __getitem__(self, idx):
		imgs = [io.imread(i) for i in self.imgs[idx][:3]]
		sample = {'images': imgs, 'label': self.imgs[idx][-1]}

		# if self.transform:
		#     sample = self.transform(sample)

		return sample

def collate_3(batch):
	data = [item['images'] for item in batch]  # just form a list of tensor
	target = [item['label'] for item in batch]
	target = torch.LongTensor(target)
	return [data, target]

train_dataset = InteractionsDataset('data/train')
val_dataset = InteractionsDataset('data/val')
test_dataset = InteractionsDataset('data/test')

train_loader = torch.utils.data.DataLoader(
			train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, collate_fn=collate_3)

model = MultiResNet()

for batch_num, (inputs, label) in enumerate(train_loader, 1):
	print(model(inputs[0][0], inputs[0][1], inputs[0][2]))
	break
























# base_transform = transforms.Compose([
# 	transforms.ToTensor(),
# 	transforms.Normalize([0.5]*3, [0.5]*3)
# 	])

# train_dataset = datasets.ImageFolder(root=train_root, transform=base_transform)
# val_dataset = datasets.ImageFolder(root=val_root, transform=base_transform)
# test_dataset = datasets.ImageFolder(root=test_root, transform=base_transform)

# def get_data_loaders(batch_size):
# 	train_loader = torch.utils.data.DataLoader(
# 			train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
# 	val_loader = torch.utils.data.DataLoader(
# 			val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
# 	return (train_loader, val_loader)

# def get_val_test_loaders(batch_size):
# 	val_loader = torch.utils.data.DataLoader(
# 			val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
# 	test_loader = torch.utils.data.DataLoader(
# 			test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
# 	return (val_loader, test_loader)

# def run():
# 	# Parameters
# 	num_epochs = 5
# 	output_period = 1
# 	batch_size = 1

# 	# setup the device for running
# 	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# 	model = resnet_18()
# 	resnet18 = models.resnet18(pretrained=True)
# 	model = model.to(device)

# 	train_loader, val_loader = dataset.get_data_loaders(batch_size)
# 	num_train_batches = len(train_loader)

# 	criterion = nn.CrossEntropyLoss().to(device)
# 	# TODO: optimizer is currently unoptimized
# 	# there's a lot of room for improvement/different optimizers
# 	optimizer = optim.SGD(model.parameters(), lr=0.55, momentum=0.4, weight_decay=1e-4)
# 	# scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

# 	epoch = 1
# 	while epoch <= num_epochs:
# 		running_loss = 0.0
# 		for param_group in optimizer.param_groups:
# 			print('Current learning rate: ' + str(param_group['lr']))
# 		# scheduler.step()
# 		model.train()

# 		for batch_num, (inputs, labels) in enumerate(train_loader, 1):
# 			inputs = inputs.to(device)
# 			labels = labels.to(device)

# 			optimizer.zero_grad()
# 			outputs = model(inputs)
# 			loss = criterion(outputs, labels)
# 			loss.backward()

# 			optimizer.step()
# 			running_loss += loss.item()

# 			if batch_num % output_period == 0:
# 				print('[%d:%.2f] loss: %.3f' % (
# 					epoch, batch_num*1.0/num_train_batches,
# 					running_loss/output_period
# 					))
# 				running_loss = 0.0
# 				gc.collect()

# 		gc.collect()
# 		# save after every epoch
# 		torch.save(model.state_dict(), "models/model.%d" % epoch)

# 		# TODO: Calculate classification error and Top-5 Error
# 		# on training and validation datasets here
# 		train_error = compute_error_rates(model, train_loader, device)
# 		valid_error = compute_error_rates(model, val_loader, device)

# 		print('Train:\tTop-1 Error: %.3f\tTop-5 Error: %.3f' % train_error)
# 		print('Valid:\tTop-1 Error: %.3f\tTop-5 Error: %.3f' % valid_error)

# 		gc.collect()
# 		epoch += 1

# def compute_error_rates(model, loader, device):
# 	model.eval()
# 	maxk = 5
# 	top_1_correct = 0.
# 	top_5_correct = 0.
# 	total = 0.

# 	for batch_num, (inputs, labels) in enumerate(loader, 1):
# 		inputs = inputs.to(device)
# 		labels = labels.to(device)
# 		outputs = model(inputs)

# 		pred = outputs.topk(maxk, 1)[1].t()
# 		correct = pred.eq(labels.view(1, -1).expand_as(pred))

# 		top_1_correct += correct[:1].view(-1).float().sum(0, keepdim=True)
# 		top_5_correct += correct[:5].view(-1).float().sum(0, keepdim=True)
# 		total += inputs.size(0)

# 	return ((1-top_1_correct/total),(1-top_5_correct/total))


# print('Starting training')
# run()
# print('Training terminated')