# import
import random
import pandas as pd
import numpy as np
import os
import re
import glob
import cv2

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
import torchvision.models as models

from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report
from tqdm.auto import tqdm

import warnings
warnings.filterwarnings(action='ignore')

device = torch.device('cuda' if torch.cuda.is_available() else torch.device('cpu'))


# Hyperparameter setting
CFG = {
    'IMG_SIZE':224,
    'EPOCHS':100,
    'LEARNING_RATE':3e-4,
    'BATCH_SIZE':32,
    'SEED':42
}


# Fixed RandomSeed
def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def seed_worker(_worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

seed_everything(CFG['SEED'])
seed_worker(CFG['SEED'])


# Data Pre-processing
all_img_list = glob.glob('/home/elicer/Jangpan/data/train/*/*')
df = pd.DataFrame(columns=['img_path', 'label'])
df['img_path'] = all_img_list
df['label'] = df['img_path'].apply(lambda x : str(x).split('/')[-2])

# stratify: 계층적 데이터 추출(클래스 분포 비율에 맞게)
train, val, _, _ = train_test_split(df, df['label'], test_size=0.3, stratify=df['label'], random_state=CFG['SEED'])

#Label-Encdoing
#preprocessing.LabelEncoder => str형 클래스를 int형으로 변환
le = preprocessing.LabelEncoder()
train['label'] = le.fit_transform(train['label'])
val['label'] = le.transform(val['label'])

#CustomDataset
class CustomDataset(Dataset):
    def __init__(self, img_path_list, label_list, transforms=None):
        self.img_path_list = img_path_list
        self.label_list = label_list
        self.transforms = transforms
    
    def __getitem__(self, index):
        img_path = self.img_path_list[index]
        image = cv2.imread(img_path)
        if self.transforms is not None:
            image = self.transforms(image=image)['image']
        if self.label_list is not None:
            label = self.label_list[index]
            return image, label
        else:
            return image
    def __len__(self):
        return len(self.img_path_list)

# always_apply == P=1.0 모든 이미지에 transform이 적용됨
train_transfrom = A.Compose([
    A.Resize(CFG['IMG_SIZE'], CFG['IMG_SIZE']),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=255.0, always_apply=False, p=1.0),
    ToTensorV2()
])
test_transform = A.Compose([
    A.Resize(CFG['IMG_SIZE'], CFG['IMG_SIZE']),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=255.0, always_apply=False, p=1.0),
    ToTensorV2()
])

train_dataset = CustomDataset(train['img_path'].values, train['label'].values, train_transfrom)
#num_workers = -1, 가능한 모든 cpu 코어 활용
#num_workers = 0 , 데이터 로딩에 대한 별도 작업자 할당 x, 메인프로세스에서 직접 데이터로드 
#                  -> 데이터로드가 오래 걸리는 경우에는 적합하지 않지만, 그렇지 않은경우 num_workers>0보다 더 빠를 수 있음
train_loader = DataLoader(train_dataset, batch_size = CFG['BATCH_SIZE'], shuffle = True, num_workers = 0)

val_dataset = CustomDataset(val['img_path'].values, val['label'].values, test_transform)
val_loader = DataLoader(val_dataset, batch_size = CFG['BATCH_SIZE'], shuffle = False, num_workers=0)


#model define
class BaseModel(nn.Module):
    def __init__(self, num_classes=len(le.classes_)):
        super(BaseModel,self).__init__()
        #self.backbone = models.efficientnet_b0(pretrained = True)
        self.backbone = models.swin_t(pretrained = True)
        self.classifier = nn.Linear(1000,num_classes)
    
    def forward(self, x):
        x = self.backbone(x)
        x = self.classifier(x)
        return x


#train/val

def train(model, optimizer, train_loader, val_loader, scheduler, device):
    model.to(device)
    criterion = nn.CrossEntropyLoss().to(device)

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

model = BaseModel()
model.eval()
optimizer = torch.optim.Adam(params = model.parameters(), lr = CFG["LEARNING_RATE"])
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor = 0.5, patience = 2, threshold_mode ='abs', min_lr = 1e-8, verbose = True)

infer_model = train(model, optimizer, train_loader, val_loader, scheduler, device)

torch.save(model, f'/home/elicer/Jangpan/weight/model.pt')
