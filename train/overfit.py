import torch
from torchvision.datasets import CIFAR10
from torchvision.transforms import v2
from IJEPA.models.vit import VIT
import torch.nn as nn
import torch.nn.functional as F

trans=v2.Compose([
            v2.ToImage(),
            v2.ToDtype(torch.float32,scale=True),
    v2.Normalize(mean=(0.4914,0.4822,0.4465),std =(0.2470,0.2435,0.2616))
])

device="mps" if torch.backends.mps.is_available() else "cpu"
dataset=CIFAR10(root='./data',train=True,transform=trans,download=True)
N=100
x=torch.stack([dataset[i][0] for i in range(N)]).to(device)
y=torch.tensor([dataset[i][1] for i in range(N)]).to(device)

encoder=VIT().to(device)
head=nn.Linear(192,10).to(device)

opt=torch.optim.AdamW(
    list(encoder.parameters()) + list(head.parameters()),
    lr=3e-4
)

for step in range(500):
    logits=head(encoder.forward_pooled(x))
    loss=F.cross_entropy(logits,y)
    opt.zero_grad()
    loss.backward()
    opt.step()
    if step%50==0:
        print(step,loss.item())

print("final", loss.item())



