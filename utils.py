
import torch
import torchvision
from carvana_data import Carvanadata
from torch.utils.data import DataLoader
import numpy as np
import os

def save(status,path):
  torch.save(status, os.path.join(path,'model.pth'))
  print("saving the model and optimizer parameters ")
  
  
def load(path,model,optimizer):
  status=torch.load(os.path.join(path,'model.pth'))
  model.load_state_dict(status["state_dict"])
  optimizer.load_state_dict(status["optimizer"])
  print("loading all trained parameters")
  
def get_loaders(
    image_dir,
    mask_dir,
    val_image_dir,
    val_mask_dir,
    batch,
    train_transform,
    val_transform,
    num_workers=4,
    pin_memory=True,
):
    train_ds = Carvanadata(
        image_dir=image_dir,
        mask_dir=mask_dir,
        transform=train_transform,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    val_ds = Carvanadata(
        image_dir=val_image_dir,
        mask_dir=val_mask_dir,
        transform=val_transform,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    return train_loader , val_loader

def check_acc(loader,model,device="cuda"):
   num_correct=0
   num_pixels=0
   dice=0
   model.eval()
   with torch.no_grad():
     for x,y in loader:
       x=x.to(device)
       y=y.to(device).unsqueeze(1).to("cpu")
       preds=torch.sigmoid(model(x))
       preds=(preds>0.5).float().to("cpu")
      
       preds=torch.flatten(preds)
       y=torch.flatten(y)

       preds=np.array(preds)
       y=np.array(y)
       
       num_correct+=(preds==y).sum()
       #num_pixels+=torch.numel(preds)
       num_pixels+=preds.size
       
       dice+=(2*((preds*y).sum()))/((preds + y).sum())
       
   accp=(num_correct/num_pixels)*100
   
       
   print(f' validation accuracy of {accp}')
   print(f'Dice score is {dice/len(loader):.2f} ')

   model.train()

def save_pred(loader, model, folder="Z:/Images/Ali/Human Liver/Nov 17/Unet/preds/"
 ,device="cuda"):
  
  model.eval()
  for idx, (x,y) in enumerate(loader):
       x=x.to(device)
       with torch.no_grad():
        preds=torch.sigmoid(model(x))
        preds=(preds>0.5).float()
        yy=y.unsqueeze(1)
       for cc in range(0,5):
         imx=x[cc,:,:]  
         imy=yy[cc,:,:]
         imp=preds[cc,:,:]
         torchvision.utils.save_image(imp, f"{folder}pred_{idx}{cc}.png")
       #  torchvision.utils.save_image(imy, f"{folder}{idx}.png")
         torchvision.utils.save_image(imx, f"{folder}x_{idx}{cc}.png")
  model.train()
































