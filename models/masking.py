import math

def sample_block(grid_size,scale_range,aspect_range,rng):
    s=rng.uniform(*scale_range)
    r=rng.uniform(*aspect_range)
    A=s*grid_size**2
    h=int(round(math.sqrt(A/r)))
    w=int(round(math.sqrt(A*r)))
    h=max(1,min(h,grid_size))
    w=max(1,min(w,grid_size))
    top=int(rng.integers(0,grid_size-h+1))
    left=int(rng.integers(0,grid_size-w+1))
    return [row*grid_size+cols
            for row in range(top,top+h)
            for cols in range(left,left+w)]

def mask_block(grid_size,n_targets,rng):
    B=[sample_block(grid_size,(0.15,0.20),(0.75,1.50),rng)
       for _ in range(n_targets)]
    C_raw=sample_block(grid_size,(0.85,1.00),(1.0,1.0),rng)
    union=set().union(*B)
    C=sorted(set(C_raw)-union)
    return B,C


