import pandas as pd
import numpy as np
import os
import cv2
import torch
import torch.nn as nn
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm

class_dict = {0: '가구수정', 1: '걸레받이수정', 2: '곰팡이', 3: '꼬임', 4: '녹오염', 5: '들뜸', 6: '면불량', 7: '몰딩수정', 8: '반점', 9: '석고수정', 10: '오염', 11: '오타공', 12: '울음', 13: '이음부불량', 14: '창틀,문틀수정', 15: '터짐', 16: '틈새과다', 17: '피스', 18: '훼손'}
CFG = {
    'IMG_SIZE':224,
    'EPOCHS':100,
    'LEARNING_RATE':3e-4,
    'BATCH_SIZE':32,
    'SEED':42
}
device = torch.device('cuda' if torch.cuda.is_available() else torch.device('cpu'))
data_folder_path = '/home/elicer/Jangpan/data'

test_transform = A.Compose([
                            A.Resize(CFG['IMG_SIZE'],CFG['IMG_SIZE']),
                            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=255.0, always_apply=False, p=1.0),
                            ToTensorV2()
                            ])

class CustomDataset(Dataset):
    def __init__(self, img_path_list, label_list, transforms=None):
        self.img_path_list = img_path_list
        self.label_list = label_list
        self.transforms = transforms
    
    def __getitem__(self, index):
        img_path = self.img_path_list[index]
        img_path = data_folder_path+img_path[1:]
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


test = pd.read_csv('/home/elicer/Jangpan/data/test.csv')
test_dataset = CustomDataset(test['img_path'].values, None, test_transform)
test_loader = DataLoader(test_dataset, batch_size=CFG['BATCH_SIZE'], shuffle=False, num_workers=0)

class BaseModel(nn.Module):
    def __init__(self, num_classes=len(class_dict)):
        super(BaseModel,self).__init__()
        #self.backbone = models.efficientnet_b0(pretrained = True)
        self.backbone = models.swin_t(pretrained = True)
        self.classifier = nn.Linear(1000,num_classes)
    
    def forward(self, x):
        x = self.backbone(x)
        x = self.classifier(x)
        return x

infer_model = torch.load("/home/elicer/Jangpan/weight/model.pt", map_location=device)


def inference(model, test_loader, device):
    model.eval()
    preds = []
    with torch.no_grad():
        for imgs in tqdm(iter(test_loader)):
            imgs = imgs.float().to(device)
            
            pred = model(imgs)
            preds += pred.argmax(1).detach().cpu().numpy().tolist()
    
    preds = [class_dict[x] for x in preds]
    return preds

preds = inference(infer_model, test_loader, device)
submit = pd.read_csv('/home/elicer/Jangpan/data/sample_submission.csv')
submit['label'] = preds
submit.to_csv('/home/elicer/Jangpan/submit/baseline_submit.csv', index=False)