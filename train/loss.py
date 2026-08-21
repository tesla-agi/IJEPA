import torch.nn.functional as F

def ijepa_loss(s_hat,s_y,tgt_idx):
    t=s_y[:,tgt_idx,:]
    t=F.layer_norm(t,t.shape[-1:])
    t=t.detach()
    return ((s_hat-t)**2).mean()

