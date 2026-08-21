import torch
import json
from torchvision import transforms,datasets
from torch.utils.data import DataLoader
from torch.optim import AdamW
from tqdm import tqdm
import os
import numpy as np
from models.vit import VIT
from models.predictor import Predictor
from utils.monitor import monitor
from train.loss import ijepa_loss
from models.masking import mask_block
from models.ema import build_target,update_target,tau_schedule

os.makedirs("./checkpoint",exist_ok=True)
if __name__=="__main__":
    device="mps" if torch.backends.mps.is_available() else "cpu"
    transform=transforms.Compose([
        transforms.RandomResizedCrop(32),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.4914,0.4822,0.4465),std =(0.2470,0.2435,0.2616))
    ])
    dataset=datasets.CIFAR10(root="./data",train=True,transform=transform,download=True)
    dataloader=DataLoader(dataset,batch_size=256,shuffle=True,num_workers=2,drop_last=True)

    online_base=VIT().to(device)
    predictor=Predictor().to(device)
    target_base=build_target(online_base).to(device)
    rng=np.random.default_rng(42)

    optimizer=AdamW(
        list(online_base.parameters())+list(predictor.parameters()),
        lr=3e-4
    )

    online_base.train()
    predictor.train()
    target_base.eval()
    loss_history=[]
    num_epochs=100
    save_every=10
    total_steps=num_epochs*len(dataloader)
    step=0
    for epoch in range(num_epochs):
        for x,_ in tqdm(dataloader,desc=f"Epoch {epoch:3d}"):
            x=x.to(device)
            B,C=mask_block(8,4,rng)
            C_t=torch.tensor(C,device=device)
            B_t=[torch.tensor(b,device=device) for b in B]
            with torch.no_grad():
                s_y=target_base(x)
            s_c=online_base(x,C_t)
            loss=0
            for B_m in B_t:
                loss=loss+ijepa_loss(predictor(s_c,C_t,B_m),s_y,B_m)
            loss=loss/len(B_t)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            tau=tau_schedule(step,total_steps)
            update_target(online_base,target_base,tau)
            loss_history.append(loss.item())
            if step%50==0:
                var,rank=monitor(s_c)
                tqdm.write(f"{step:5d}  loss {loss.item():.4f}  var {var:.4f}  rank {rank:6.1f}")
            step+=1
        if (epoch+1) % save_every == 0:
            torch.save(online_base.state_dict(), f"./checkpoint/vit_on{epoch + 1}.pth")
            torch.save(target_base.state_dict(), f"./checkpoint/vit_tgt{epoch + 1}.pth")
            tqdm.write(f"checkpoint saved · epoch {epoch + 1} · step {step}")
    torch.save(online_base.state_dict(), "./checkpoint/vit_on.pth")
    torch.save(target_base.state_dict(), "./checkpoint/vit_tgt.pth")
    tqdm.write(f"final checkpoint saved · {step} steps")
    with open(f"./checkpoint/loss_history.json", 'w') as f:
        json.dump(loss_history, f)