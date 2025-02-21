import numpy as np
import torch
from MyTSPDataSet import MyTSPDataSet
from TSPLibDataSet import TSPLibDataSet
from torch.utils.data import DataLoader
import gzip
import math
import matplotlib.pyplot as plt
import os

def cycle(loader):
    while True:
        for data in loader:
            yield data

def get_loaders_TSP(filename, seq_len, batch_size):
 
    train_dataset = MyTSPDataSet(filename,seq_len)
    val_dataset   = MyTSPDataSet(filename,seq_len)
    train_loader  = DataLoader(train_dataset, shuffle=True, batch_size = batch_size)
    val_loader    = DataLoader(val_dataset, batch_size = batch_size)
    return train_loader, val_loader, val_dataset

def get_loaders_TSPLib(filename, seq_len, batch_size):
 
    gen_dataset = TSPLibDataSet(filename, seq_len)
    gen_loader  = cycle(DataLoader(gen_dataset, batch_size = batch_size))
    return gen_loader

def compute_distance_matrix(node_list): 
    sz = len(node_list) + 1 
    dmatrix = np.zeros((sz,sz)) 
    for i in range (sz-1):
        for j in range(sz-1):
            bb = (node_list[i].X - node_list[j].X)**2
            cc = (node_list[i].Y - node_list[j].Y)**2
            dmatrix[i+1,j+1] = math.sqrt((node_list[i].X - node_list[j].X)**2  + (node_list[i].Y - node_list[j].Y)**2)
    return dmatrix

def plotTSP(tour, node_data, num_iters=1):

    """
    path: List of lists with the different orders in which the nodes are visited
    points: coordinates for the different nodes
    num_iters: number of paths that are in the path list
    
    """

    node_d = node_data.detach().cpu()
    x = []; y = []
    for i in tour:
        x.append(node_d[0,i,1])
        y.append(node_d[0,i,2])
    
    plt.plot(x, y, 'co')


    a_scale = float(max(x))/float(100)

    if num_iters > 1:

        for i in range(1, num_iters):

            xi = []; yi = [];
            for j in paths[i]:
                xi.append(points[j][0])
                yi.append(points[j][1])

            plt.arrow(xi[-1], yi[-1], (xi[0] - xi[-1]), (yi[0] - yi[-1]), 
                    head_width = a_scale, color = 'r', 
                    length_includes_head = True, ls = 'dashed',
                    width = 0.001/float(num_iters))
            for i in range(0, len(x) - 1):
                plt.arrow(xi[i], yi[i], (xi[i+1] - xi[i]), (yi[i+1] - yi[i]),
                        head_width = a_scale, color = 'r', length_includes_head = True,
                        ls = 'dashed', width = 0.001/float(num_iters))


    plt.arrow(x[-1], y[-1], (x[0] - x[-1]), (y[0] - y[-1]), head_width = a_scale, 
            color ='g', length_includes_head=True)
    for i in range(0,len(x)-1):
        plt.arrow(x[i], y[i], (x[i+1] - x[i]), (y[i+1] - y[i]), head_width = a_scale,
                color = 'g', length_includes_head = True)


    plt.xlim(0, max(x)*1.1)
    plt.ylim(0, max(y)*1.1)
    plt.show()
    
def save_model(model, i, optim, fname):
    print("----------saving model-----------------")
    checkpoint_data = {
    'epoch': i,
    'state_dict': model.state_dict(),
    'optimizer': optim.state_dict()
    }
    ckpt_path = os.path.join("checkpoint/" + fname) 
    torch.save(checkpoint_data, ckpt_path)
    model.train()