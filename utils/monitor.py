import torch

def monitor(s_c):
    with torch.no_grad():
        s_c=s_c.flatten(0,1)
        var=s_c.var(dim=0).mean()
        sigma=torch.linalg.svdvals(s_c.cpu())
        p=sigma/(sigma.sum()+1e-12)
        entropy=-torch.sum(p*torch.log(p+1e-12))
        eff_rank=torch.exp(entropy)
    return var.item(),eff_rank.item()

