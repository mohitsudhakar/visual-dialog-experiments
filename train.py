
import os
import yaml
import argparse
from tqdm import tqdm
import torch
from torch import nn
import torchtext
import torchvision.transforms as transforms
from data.dataset import VisDialDataset
from torch.utils.data import DataLoader
from torch.optim import Adam, lr_scheduler

from models.encdec import EncoderDecoder

# parser = argparse.ArgumentParser()
# parser.add_argument('--config', type=str, default='configs/ls_disc_vgg16.yml', help='Configs path')
# args = parser.parse_known_args()

config = yaml.load(open('configs/lf_disc_vgg16.yml'))
return_options = True if config['model']['decoder'] == 'disc' else False

# Init datasets
trainset = VisDialDataset(config=config['dataset'],
                          dialogs_jsonpath='data/visdial_1.0_train.json',
                          return_options=return_options,
                          add_boundary_toks=True,
                          overfit=True)
valset = VisDialDataset(config=config['dataset'],
                        dialogs_jsonpath='data/visdial_1.0_val.json',
                        dense_annotations_jsonpath='data/visdial_1.0_val_dense_annotations.json',
                        return_options=return_options,
                        add_boundary_toks=True,
                        overfit=True)

# Init data loaders
train_loader = DataLoader(trainset,
                          batch_size = config['solver']['batch_size'],
                          num_workers = 5,
                          shuffle = True)
val_loader = DataLoader(valset,
                          batch_size = config['solver']['batch_size'],
                          num_workers = 5,
                          shuffle = False)


model = EncoderDecoder(config["model"], trainset.vocabulary)

optimizer = Adam(model.parameters(), lr=config["solver"]["initial_lr"]) # todo: diff b/t Adam and Adamax?
scheduler = lr_scheduler.CyclicLR(optimizer)    # todo: change sched

# Loss function.
if config["model"]["decoder"] == "disc":
    criterion = nn.CrossEntropyLoss()
elif config["model"]["decoder"] == "gen":
    criterion = nn.CrossEntropyLoss(ignore_index=trainset.vocabulary.PAD_INDEX)
else:
    raise NotImplementedError

# todo: Add SummaryWriter

num_epochs = config['solver']['num_epochs']
it = 0
for epoch in range(num_epochs):

    print('Epoch', epoch)

    for i, batch in enumerate(tqdm(train_loader)):

        for key in ['img_feat', 'ques', 'hist', 'ans_ind', 'ans_out']:
            batch[key] = batch[key].cuda()

        if config["model"]["decoder"] == "disc":
            target = batch['ans_ind']
        else:
            target = batch['ans_out']

        optimizer.zero_grad()
        output = model(batch['img_feat'],
                       batch['ques'],
                       batch['hist'])

        loss = criterion(output.view(-1, output.size(-1)), target.view(-1))
        loss.backward()
        optimizer.step()
        scheduler.step(it)
        it += 1

    # todo: Validation step
    # Note; validation returns list of options, but training returns one answer,
    # Evaluation occurs on validation (sorting list of options) - ranking task

