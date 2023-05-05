import argparse
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
from metric import metric, plot_confusion_matrix
from others import *

from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.metrics import f1_score,classification_report, confusion_matrix, accuracy_score
from tqdm.auto import tqdm
import yaml
import wandb
import warnings
warnings.filterwarnings(action='ignore')

device = torch.device('cuda' if torch.cuda.is_available() else torch.device('cpu'))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--yaml_path', type=str, default = 'Baseyaml.yaml', help = 'yaml 파일 위치')
    args = parser.parse_args()

# yaml 파일 가져오기 및 실험 기본 설정
with open(args.yaml_path) as f:
    CFG = yaml.load(f, Loader=yaml.FullLoader)
folder_path = createDirectory(os.path.join(CFG['ROOT_PATH'], CFG['NAME']))
yalm_string = yaml.dump(CFG)
CFG['NAME'] = folder_path.split('/')[-1]
with open(os.path.join(folder_path, CFG['NAME']+'.yaml'), 'w') as outfile:
    yaml.dump(CFG, outfile)

# wandb 설정
wandb.init(
    project='jangpan',
    name=CFG['NAME'],
    config=CFG
    )

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
    criterion = create_criterion(CFG['LOSS'], reduction="none").to(device)

    best_score = 0
    best_model = None

    for epoch in range(1, CFG['EPOCHS']+1):
        model.train()
        train_losses = []
        train_predictions = []
        train_labels = []
        
        for imgs, labels in tqdm(iter(train_loader)):
            imgs = imgs.float().to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            output = model(imgs)
            losses = criterion(output, labels)
            losses.mean().backward()
            optimizer.step()
            
            train_losses.append(losses.detach().cpu())
            train_predictions.append(torch.argmax(output, dim=1))
            train_labels.append(labels)
        
        train_predictions = torch.cat(train_predictions)
        train_labels = torch.cat(train_labels)
        train_losses = torch.cat(train_losses)

        if epoch % CFG['LOG_INTERVAL'] == 0 or epoch == CFG['EPOCHS']:
            train_metric = metric(train_losses, train_predictions, train_labels, len(le.classes_))
            val_metric = validation(model, criterion, val_loader, device)

            print(f'Epoch [{epoch}], Train Loss : [{train_metric["total_loss"]:.5f}] Val Loss : [{val_metric["total_loss"]:.5f}] Val Weighted F1 Score : [{val_metric["weighted_f1"]:.5f}]')
            
            label_dict = dict(zip(le.transform(le.classes_), le.classes_))
            train_log_dict = {"Train/ Loss": train_metric["total_loss"], "Train/ ACC": train_metric["total_accuracy"],"Train/ Weighted F1 Score": train_metric["weighted_f1"]}
            for index, label in label_dict.items():
                train_log_dict[f"Train/ ({label}) Loss"] = train_metric["class_loss"][index]
                train_log_dict[f"Train/ ({label}) ACC"] = train_metric["class_accuracy"][index]
                train_log_dict[f"Train/ ({label}) F1 score"] = train_metric["class_f1"][index]
            wandb.log(train_log_dict, step=epoch)

            val_log_dict = {"Val/ Loss": val_metric["total_loss"], "Val/ ACC": val_metric["total_accuracy"],"Val/ Weighted F1 Score": val_metric["weighted_f1"]}
            for index, label in label_dict.items():
                val_log_dict[f"Val/ ({label}) Loss"] = val_metric["class_loss"][index]
                val_log_dict[f"Val/ ({label}) ACC"] = val_metric["class_accuracy"][index]
                val_log_dict[f"Val/ ({label}) F1 score"] = val_metric["class_f1"][index]
            wandb.log(val_log_dict, step=epoch)

            wandb.log({"lr": optimizer.param_groups[0]['lr'], "epoch": epoch}, step =epoch)
            if scheduler is not None:
                scheduler.step(train_metric["weighted_f1"])

            if best_score < train_metric["weighted_f1"]:
                best_score = train_metric["weighted_f1"]
                best_model = model
                plot_confusion_matrix(val_metric["cmatrix"], folder_path)


    return best_model

def validation(model, criterion, val_loader, device):
    model.eval()
    val_losses = []
    val_predictions = []
    val_labels = []

    with torch.no_grad():
        for imgs, labels in tqdm(iter(val_loader)):
            imgs = imgs.float().to(device)
            labels = labels.to(device)

            pred = model(imgs)
            loss = criterion(pred, labels)

            val_losses.append(loss.detach().cpu())
            val_predictions.append(torch.argmax(pred, dim=1))
            val_labels.append(labels)
            
        val_predictions = torch.cat(val_predictions)
        val_labels = torch.cat(val_labels)
        val_losses = torch.cat(val_losses)

        val_metric = metric(val_losses, val_predictions, val_labels, len(le.classes_))
    return val_metric


model = create_model(CFG['MODEL'], num_classes=len(le.classes_))
optimizer = create_optimizer(CFG['OPTIMIZER'], params = model.parameters(), lr = CFG["LEARNING_RATE"])
scheduler =  create_scheduler(CFG['SCHEDULER'], optimizer = optimizer)
infer_model = train(model, optimizer, train_loader, val_loader, scheduler, device)
torch.save(infer_model, os.path.join(folder_path,'model.pt'))
wandb.finish()
