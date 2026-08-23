import torch
import torch.nn as nn
import torch.nn.functional as F
from models.vit import VIT
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision import datasets,transforms

def train_probe(encoder,classifier,train_loader,optimizer,device):
    classifier.train()
    total=0
    for img,label in train_loader:
        img,label=img.to(device),label.to(device)
        with torch.no_grad():
            features=encoder.forward_pooled(img)
        logits=classifier(features)
        loss=F.cross_entropy(logits,label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total+=loss.item()
    return total/len(train_loader)

def evaluate(encoder,classifier,test_loader,device):
    encoder.eval()
    classifier.eval()
    correct=0
    total=0
    with torch.no_grad():
        for img,label in test_loader:
            img,label=img.to(device),label.to(device)
            features=encoder.forward_pooled(img)
            logits=classifier(features)
            pred=torch.argmax(logits,dim=1)
            correct+=(pred==label).sum().item()
            total+=label.size(0)

        return 100*correct/total


if __name__=="__main__":
    device="mps" if torch.backends.mps.is_available() else "cpu"
    enc=VIT().to(device)
    enc.load_state_dict(torch.load("./checkpoint/abl_b_on40.pth",map_location=device))
    for p in enc.parameters():
        p.requires_grad=False
    enc.eval()
    classifier=nn.Linear(192,10).to(device)
    optimizer=Adam(classifier.parameters(),lr=1e-3)

    tfm1=transforms.Compose([
        transforms.RandomCrop(32,padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.4914,0.4822,0.4465),std =(0.2470,0.2435,0.2616))
    ])

    tfm2=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.4914, 0.4822, 0.4465),std=(0.2470, 0.2435, 0.2616))
    ])

    train_dataset=datasets.CIFAR10(root='./data',transform=tfm1,train=True,download=True)
    test_dataset=datasets.CIFAR10(root='./data',transform=tfm2,train=False,download=True)
    train_loader=DataLoader(train_dataset,batch_size=256,shuffle=True,num_workers=2)
    test_loader=DataLoader(test_dataset,batch_size=256,shuffle=False,num_workers=2)

    for epoch in range(15):
        loss=train_probe(enc,classifier,train_loader,optimizer,device=device)
        accuracy=evaluate(enc,classifier,test_loader,device)
        print(f"epoch {epoch:2d}  loss {loss:.4f}  acc {accuracy:.2f}")
