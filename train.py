# import
import random
import pandas as pd
import numpy as np
import os
import glob

import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

from data_loaders import create_dataset, create_train_augmentation, create_test_augmentation
from model import create_model
from optimizer import create_optimizer
from scheduler import create_scheduler
from loss import create_criterion
from others import *

from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report
from tqdm.auto import tqdm
import yaml
import warnings
warnings.filterwarnings(action='ignore')

device = torch.device('cuda' if torch.cuda.is_available() else torch.device('cpu'))

# yaml 파일 가져오기 및 실험 기본 설정
with open('Baseyaml.yaml') as f:
    CFG = yaml.load(f, Loader=yaml.FullLoader)
folder_path = createDirectory(os.path.join(CFG['ROOT_PATH'], CFG['NAME']))
yalm_string = yaml.dump(CFG)
CFG['NAME'] = folder_path.split('/')[-1]
with open(os.path.join(folder_path, CFG['NAME']+'.yaml'), 'w') as outfile:
    yaml.dump(CFG, outfile)



# 재현성
seed_everything(CFG['SEED'])
seed_worker(CFG['SEED'])

# Data Pre-processing
all_img_list = glob.glob('/home/elicer/Jangpan/data/train/*/*')
df = pd.DataFrame(columns=['img_path', 'label'])
df['img_path'] = all_img_list
df['label'] = df['img_path'].apply(lambda x : str(x).split('/')[-2])

train, val, _, _ = train_test_split(df, df['label'], test_size=0.3, stratify=df['label'], random_state=CFG['SEED'])

#Label-Encdoing
le = preprocessing.LabelEncoder()
train['label'] = le.fit_transform(train['label'])
val['label'] = le.transform(val['label'])


# transform / dataset / dataloader /sampler
train_transform = create_train_augmentation(CFG['TRAIN_AUGMENTATION'], resize = CFG['IMG_SIZE'])
test_transform = create_test_augmentation(CFG['TEST_AUGMENTATION'], resize = CFG['IMG_SIZE'])

weights = make_weights(train['label'], len(le.classes_))
weights = torch.DoubleTensor(weights)
sampler = torch.utils.data.sampler.WeightedRandomSampler(weights, len(weights))

train_dataset = create_dataset(
    CFG['DATASET'],
    img_path_list = train['img_path'].values,
    label_list =  train['label'].values,
    transforms = train_transform)
train_loader = DataLoader(train_dataset, batch_size = CFG['BATCH_SIZE'], num_workers = 0, sampler = sampler)

val_dataset = create_dataset(
    CFG['DATASET'],
     img_path_list = val['img_path'].values,
     label_list =  val['label'].values,
     transforms = test_transform)
val_loader = DataLoader(val_dataset, batch_size = CFG['BATCH_SIZE'], shuffle = False, num_workers=0)


#train/val
def train(model, optimizer, train_loader, val_loader, scheduler, device):
    model.to(device)
    criterion = create_criterion(CFG['LOSS']).to(device)

    best_score = 0
    best_model = None

    for epoch in range(1, CFG['EPOCHS']+1):
        model.train()
        train_loss = []
        for imgs, labels in tqdm(iter(train_loader)):
            imgs = imgs.float().to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            output = model(imgs)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()

            train_loss.append(loss.item())

        if epoch % CFG['LOG_INTERVAL'] == 0 or epoch == CFG['EPOCHS']:
            _val_loss, _val_score = validation(model, criterion, val_loader, device)
            _train_loss = np.mean(train_loss)
            print(f'Epoch [{epoch}], Train Loss : [{_train_loss:.5f}] Val Loss : [{_val_loss:.5f}] Val Weighted F1 Score : [{_val_score:.5f}]')

            if scheduler is not None:
                scheduler.step(_val_score)

            if best_score < _val_score:
                best_score = _val_score
                best_model = model

    return best_model

def validation(model, criterion, val_loader, device):
    model.eval()
    val_loss = []
    preds, true_labels = [], []

    with torch.no_grad():
        for imgs, labels in tqdm(iter(val_loader)):
            imgs = imgs.float().to(device)
            labels = labels.to(device)

            pred = model(imgs)
            loss = criterion(pred, labels)

            preds += pred.argmax(1).detach().cpu().numpy().tolist()
            true_labels += labels.detach().cpu().numpy().tolist()

            val_loss.append(loss.item())

        _val_loss = np.mean(val_loss)
        _val_score = f1_score(true_labels, preds, average = 'weighted')
    return _val_loss, _val_score


model = create_model(CFG['MODEL'], num_classes=len(le.classes_))
optimizer = create_optimizer(CFG['OPTIMIZER'], params = model.parameters(), lr = CFG["LEARNING_RATE"])
scheduler =  create_scheduler(CFG['SCHEDULER'], optimizer = optimizer)
infer_model = train(model, optimizer, train_loader, val_loader, scheduler, device)
torch.save(infer_model, os.path.join(folder_path,'model.pt'))
