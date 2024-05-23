import time
import cv2
import torch
from tqdm import tqdm
from torch.utils.data import Dataset
from dataloader import default_collate, batch2device

class CustomDS(Dataset):
    def __init__(self) -> None:
        super().__init__()
    
    def  __len__(self):
        return 1000
    
    def __getitem__(self, idx):
        img = cv2.imread('images/sea.jpg')
        img = cv2.resize(img, (640, 360))
        img = torch.from_numpy(img).to(float)
        img = img + torch.randn_like(img)
        return img


def torch_loader(dataset):
    from torch.utils.data import DataLoader
    loader = DataLoader(dataset, batch_size=16, num_workers=8, collate_fn=default_collate, pin_memory=True)
    for dl in loader:
        dl = dl.cuda()
        pass
    
def mini_lodaer(dataset):
    from dataloader import DataLoader
    loader = DataLoader(dataset, batch_size=16, num_workers=8, collate_fn=default_collate, 
                        preload=True, preload_device='cuda:0', preproc_fun=batch2device)
    for dl in loader:
        pass


st = time.time()
for i in tqdm(range(5)):
    torch_loader(CustomDS())
print('torch time:', time.time() - st)

st = time.time()
for i in tqdm(range(5)):
    mini_lodaer(CustomDS())
print('mini time:', time.time() - st)