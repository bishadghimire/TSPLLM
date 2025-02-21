import torch
import sys
from torch.utils.data import Dataset
from mydataclasses.OptimizationResult import OptimizationResult
from mydataclasses.Node import Node
import json
import numpy as np
import Utils

class MyTSPDataSet(Dataset):
    def __init__(self, filename, seq_len):
        super().__init__()

        fstr = open(filename)
        all_data = json.load(fstr)
        count = len(all_data)
        opt_data = [OptimizationResult(d['OrigNodeList'],
            d['OptimizedNodeList'],
            d['TourCost']) for d in all_data]

        for doptr in opt_data: 
            doptr.OrigNodeList = [Node(d['NodeNum'],d['X'],d['Y']) for d in doptr.OrigNodeList]
        for doptr in opt_data:  
            doptr.OptimizedNodeList = [Node(d['NodeNum'],d['X'],d['Y']) for d in doptr.OptimizedNodeList]
        yy = opt_data[0]
        orig_len = len(yy.OrigNodeList) 
        opt_len = len(yy.OptimizedNodeList) 
        self.data = np.zeros((len(opt_data),orig_len+opt_len,orig_len+3))
        for k in range(len(opt_data)):
            dmatrix = Utils.compute_distance_matrix(opt_data[k].OrigNodeList)
            count = 0
            for i in range(len(opt_data[k].OrigNodeList)):
                self.data[k,i,0] = opt_data[k].OrigNodeList[i].NodeNum
                self.data[k,i,1] = opt_data[k].OrigNodeList[i].X
                self.data[k,i,2] = opt_data[k].OrigNodeList[i].Y
                self.data[k,i,3:] = dmatrix[i+1,1:]
            count = len(opt_data[k].OrigNodeList)
            count_opt_tour = len(opt_data[k].OptimizedNodeList)
            for i in range(0,count_opt_tour):
                j = i
                self.data[k,count,:] = self.data[k,opt_data[k].OptimizedNodeList[i].NodeNum-1,:]
                count = count + 1
        self.data[:,:,0] = self.data[:,:,0]-1  

        self.masked_output = np.ones((self.data.shape[0],orig_len,orig_len))
        for i in range(0,self.data.shape[0]): 
            for j in range(0,orig_len):
                self.masked_output[i,j,0] = 0   

                for k in range(0,j):
                    self.masked_output[i,j,self.data[i,k+orig_len+1,0].astype(int)] = 0 

        self.seq_len = seq_len

    def __getitem__(self, index):
        return torch.tensor(self.data[index]).float().cuda(),torch.tensor(self.masked_output[index]).cuda()

    def __len__(self):
        return self.data.shape[0] 

    
def main():
    t1 = MyTSPDataSet(512)

if __name__ == "__main__":
    sys.exit(int(main() or 0))