import argparse
import pandas as pd
import numpy as np
import os
import torch

from data_loaders import create_dataset, create_test_augmentation
from model import create_model
from others import *


from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm
import yaml

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('yaml_path', type=str, help = 'yaml 파일 위치')
    args = parser.parse_args()

# yaml 파일 가져오기 및 실험 기본 설정
with open(args.yaml_path) as f:
    CFG = yaml.load(f, Loader=yaml.FullLoader)
folder_path = os.path.join(CFG['ROOT_PATH'], CFG['NAME'])
device = torch.device('cuda' if torch.cuda.is_available() else torch.device('cpu'))
class_dict = {0: '가구수정', 1: '걸레받이수정', 2: '곰팡이', 3: '꼬임', 4: '녹오염', 5: '들뜸', 6: '면불량', 7: '몰딩수정', 8: '반점', 9: '석고수정', 10: '오염', 11: '오타공', 12: '울음', 13: '이음부불량', 14: '창틀,문틀수정', 15: '터짐', 16: '틈새과다', 17: '피스', 18: '훼손'}

# transform / dataset / dataloader 
test = pd.read_csv('/home/elicer/Jangpan/data/test.csv')
test['img_path'] = test['img_path'].apply(lambda x : os.path.join('/home/elicer/Jangpan/data',*x.split('/')[1:]))
print(test['img_path'])
test_transform = create_test_augmentation(CFG['TEST_AUGMENTATION'], resize = CFG['IMG_SIZE'])
test_dataset = create_dataset(
    CFG['DATASET'],
    img_path_list = test['img_path'].values,
    label_list =  None,
    transforms = test_transform
)
test_loader = DataLoader(test_dataset, batch_size=CFG['BATCH_SIZE'], shuffle=False, num_workers=0)

model = create_model(CFG['MODEL'], num_classes=len(class_dict))
infer_model = torch.load(os.path.join(CFG['ROOT_PATH'], CFG['NAME'], 'model.pt'), map_location=device)


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
submit.to_csv(os.path.join(CFG['ROOT_PATH'], CFG['NAME'], 'submit.csv'), index=False)


