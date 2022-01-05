
import utils
import Unet_tutorial
import carvana_data
import torch
import albumentations as A
import albumentations
import torch.nn as nn
import torch.optim as optim
from Unet_tutorial import Unet
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
from utils import get_loaders, check_acc, save_pred, save, load
import torchvision.transforms.functional as TF

device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)
name=torch.cuda.get_device_name(device)
print(f'our cuda device is {name}')

### Hyper paramters
lrate=0.0003
epoch=10

batch=5
img_w= 675  #1353
img_h= 333  ## 667
need_load=False
num_workers=2
pin_memory=True


path='C:/Users/sem/Documents/Ali/Unet_model/checkpoint/'
image_dir='Z:/Images/Ali/Human Liver/Nov 17/Unet/training_images/'
mask_dir='Z:/Images/Ali/Human Liver/Nov 17/Unet/training_mask_r/'

val_image_dir='Z:/Images/Ali/Human Liver/Nov 17/Unet/validation/'
val_mask_dir='Z:/Images/Ali/Human Liver/Nov 17/Unet/validation_mask/'


def train_fn(loader, model, optimizer, loss_fn,scaler):
    loop=tqdm(loader)
    
    for batch_idx, (data, target) in enumerate(loop):
        data=data.to(device)
        target=target.float().unsqueeze(1).to(device)
        target=target.float().to(device)
       
        #### forward pass
        with torch.cuda.amp.autocast():
          prediction=model(data)
          loss=loss_fn(prediction,target)
       ### backward pass
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
       # scaler.update()
        
        loop.set_postfix(loss=loss.item())
        
def main():
    train_transform=A.Compose(
        [
            A.Resize(height=img_h,width=img_w),
            A.Rotate(limit=30,p=0.2),
            A.HorizontalFlip(p=0.4),
            A.VerticalFlip(p=0.1),
            A.Normalize(
                mean=[0],
                std=[1],
                max_pixel_value=255,
                ),
            ToTensorV2(),
            
        
        ])

    val_transform=A.Compose(
        [
            A.Resize(height=img_h,width=img_w),
            A.Normalize(
                mean=[0],
                std=[1],
                max_pixel_value=255,
                ),
            ToTensorV2(),
            
        
        ],

        )
          

    model=Unet(in_channel=1, out_channel=1).to(device)
    loss_fn=nn.BCEWithLogitsLoss()
    optimizer=optim.Adam(model.parameters(),lr=lrate)
    
    train_loader, val_loader = get_loaders(
       image_dir,
       mask_dir,
       val_image_dir,
       val_mask_dir,
       batch,
       train_transform,
       val_transform,
       num_workers=num_workers,
       pin_memory=pin_memory
   )
   
  
    scaler = torch.cuda.amp.GradScaler()

    if need_load:
        load(path, model,optimizer)
    
    for items in range(epoch):
       train_fn(train_loader, model, optimizer, loss_fn,scaler)
       print(f'running epoch # {items}')
       check_acc(val_loader,model,device="cuda")
      
       
       save_pred(val_loader,model,folder="Z:/Images/Ali/Human Liver/Nov 17/Unet/preds/")
       checkpoint = {
            "state_dict": model.state_dict(),
            "optimizer":optimizer.state_dict(),
        }
       save(checkpoint,path)

if __name__=="__main__":
   main()      
        

