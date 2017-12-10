import os
import matplotlib.pyplot as plt

def heatmap_2D(x):
    
    x = x.squeeze()

    # Currently only shows in pop-up
    plt.imshow(x, cmap='magma', interpolation='nearest')
    return plt

def heatmap_3D(x):
    raise NotImplementedError

def heatmap_4D(x):
    # 4D images only        

    X = x.shape[3]  
    Y = x.shape[2]
    Z = x.shape[1]
    A = x.shape[0]
    
    print(X,Y,Z,A)
    
    # Plot Ic x Oc for each Kernel 
    fig, ax = plt.subplots(nrows=X, ncols=Y)
    if X == 1 and Y == 1:        
        plt.imshow(x[:, :, 0, 0], cmap='hot', interpolation='nearest')
    else:
        for xx,row in enumerate(ax):
            for yy,col in enumerate(row):
                col.pcolormesh(x[:,:,xx,yy])
                   
    return plt
    
    
def heatmap_unimplemented(x):
    raise NotImplementedError

    
def heatmap_weights(x, label, directory, args):

    dims = 0
    for y in x.shape:
        if y > 1:
            dims += 1

    if dims == 2:
        plt = heatmap_2D(x)
    elif dims == 3:
        plt = heatmap_3D(x)
    elif dims == 4:        
        plt = heatmap_4D(x)
    else:
        heatmap_unimplemented(x)

    plt.savefig(directory+"/"+label.replace("/","__")+"_heatmap.svg", format="svg")
    plt.close()
        

