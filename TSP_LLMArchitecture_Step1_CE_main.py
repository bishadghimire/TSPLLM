import random
import tqdm
import numpy as np
import os

import torch
import torch.optim as optim
from AutoRegressiveWrapper import AutoRegressiveWrapper
from models.SimpleTransformer import SimpleTransformer
import Utils
import sys
import math
from models.PerceiverAR import PerceiverARTransformer


NUM_EPOCHS = int(25)
BATCH_SIZE = 16
GRADIENT_ACCUMULATE_EVERY = 1
NUM_NODES = 29 
LEARNING_RATE = 1e-4  
VALIDATE_EVERY  = 1000
GENERATE_EVERY  = 300  
GENERATE_LENGTH = NUM_NODES 
SEQ_LENGTH = NUM_NODES * 2 
                           
EMBEDDING_SIZE = 192 
NUM_LAYERS = 12
NUM_HEADS = 6
LATENT_LEN = NUM_NODES 
RESUME_TRAINING = False

TrainDataset_File = "data/TSPTestData_for_Rand29Nodes_1000.txt" 
TSPLibDataset_File = "data/Bays29_Test_Opt9076.txt" 
ValidationDataset_File = "data/TSPValidatinData_for_Nodes29_2.txt" 
SAVE_FILE_NAME = "TSPModel_" + str(NUM_LAYERS) + "_" + str(NUM_HEADS) + "_" + str(EMBEDDING_SIZE) + \
            "_" + "Nodes" + str(NUM_NODES) + "_Bays29_STEP1.pt"


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

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def configure_optimizers(mymodel):
    """
    We are separating out all parameters of the model into two buckets: those that will experience
    weight decay for regularization and those that won't (biases, and layernorm/embedding weights).
    Retur the PyTorch optimizer object.
    """

    decay = set()
    no_decay = set()
    whitelist_weight_modules = (torch.nn.Linear, )
    blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding)
    for mn, m in mymodel.named_modules():
        for pn, p in m.named_parameters():
            fpn = '%s.%s' % (mn, pn) if mn else pn 

            if pn.endswith('bias'):
                no_decay.add(fpn)
            elif pn.endswith('weight') and isinstance(m, whitelist_weight_modules):
                decay.add(fpn)
            elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules):
                no_decay.add(fpn)

    param_dict = {pn: p for pn, p in mymodel.named_parameters()}
    inter_params = decay & no_decay
    union_params = decay | no_decay
    assert len(inter_params) == 0, "parameters %s made it into both decay/no_decay sets!" % (str(inter_params), )
    assert len(param_dict.keys() - union_params) == 0, "parameters %s were not separated into either decay/no_decay set!" \
                                                % (str(param_dict.keys() - union_params), )

    optim_groups = [
        {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": 0.1},
        {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
    ]
    optimizer = torch.optim.AdamW(optim_groups, lr=LEARNING_RATE, betas=(0.9,0.95))
    return optimizer

def compute_tour_cost(x, generate_len):  
    cost = 0
    curr_node_x = x[:,generate_len,1].item()
    curr_node_y = x[:,generate_len,2].item()
    for i in range(1,generate_len+1):
        next_node_x = x[:,generate_len+i,1].item()
        next_node_y = x[:,generate_len+i,2].item()
        dist = math.sqrt((curr_node_x - next_node_x)**2 + (curr_node_y - next_node_y)**2)
        cost = cost + dist
        curr_node_x = next_node_x
        curr_node_y = next_node_y
    return cost

def validate_model(model, x_val, masked_output_val, opt_cost_val):
    total_percent_cost = 0
    for i in range(0,x_val.shape[0]):
        inp_pad_val = torch.zeros((1,SEQ_LENGTH+1,x_val.shape[2])).cuda()
        inp_pad_val[:,0:GENERATE_LENGTH+1,:] = x_val[i:i+1,0:GENERATE_LENGTH+1,:]
        tour_val, cost_val, valid = model.generate(inp_pad_val, GENERATE_LENGTH)
        percent_optimal = ((cost_val - opt_cost_val[i])/opt_cost_val[i])*100
        total_percent_cost = total_percent_cost + ((cost_val - opt_cost_val[i])/opt_cost_val[i])*100
    return total_percent_cost/x_val.shape[0]

def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
 
    PAR_model = PerceiverARTransformer(
        dim = EMBEDDING_SIZE, 
        num_unique_tokens = NUM_NODES,  
        num_layers = NUM_LAYERS, 
        heads = NUM_HEADS, 
        sequence_len = SEQ_LENGTH,
        latent_len = LATENT_LEN
     ).cuda()

    model = AutoRegressiveWrapper(PAR_model,latent_len=LATENT_LEN)
    
    model.cuda()
    pcount = count_parameters(model)
    print("count of parameters in the model = ", pcount, " million")
    
    gen_loader = Utils.get_loaders_TSPLib(TSPLibDataset_File,200,1) 
    gen_loader_val = Utils.get_loaders_TSPLib(ValidationDataset_File,200,10)

    train_loader, val_loader, val_dataset = Utils.get_loaders_TSP(TrainDataset_File, SEQ_LENGTH, BATCH_SIZE)
 
    optim_RL = torch.optim.Adam(model.parameters(), lr = LEARNING_RATE)     
    optim = configure_optimizers(model)

    valid_x,_, _ = next(gen_loader)
       
    best_cost = 999999
    best_avg_percent_optimal = 500
    

    if RESUME_TRAINING == False:
        start = 0
    else:
        checkpoint_data = torch.load('checkpoint/' + SAVE_FILE_NAME)
        model.load_state_dict(checkpoint_data['state_dict'])
        start = checkpoint_data['epoch']
        optim.load_state_dict(checkpoint_data['optimizer'])
        for param_group in optim.param_groups:  
           param_group['lr'] = LEARNING_RATE
        
        start = checkpoint_data['epoch']
        model.eval()
        inp = valid_x
        inp_pad = torch.zeros((inp.shape[0],SEQ_LENGTH+1,inp.shape[2])).cuda()
        inp_pad[:,0:GENERATE_LENGTH+1,:] = inp[:,0:GENERATE_LENGTH+1,:]
        tour, cost, valid = model.generate(inp_pad, GENERATE_LENGTH)
        best_cost = cost
        print("----------generated output-----------")
        print('COST =',cost, ' tour=',tour)
        print("----------end generated output-----------")
        Utils.plotTSP(tour, inp,1)
        return
 
    x_val, masked_output_val, opt_cost_val = next(gen_loader_val) 
    for epoch in range(start, NUM_EPOCHS): 
        for i, data in enumerate(train_loader):    
            model.train()
            total_loss = 0
            x, masked_output = data
            loss = model(x, masked_output)
            loss.backward()
            if (i%100 == 0):
                print(f'training loss: {loss.item()} -- iteration = {i}')

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()
            optim.zero_grad()
        
            if i % GENERATE_EVERY == 0: 
                model.eval()
                inp = valid_x
                inp_pad = torch.zeros((inp.shape[0],SEQ_LENGTH+1,inp.shape[2])).cuda()
                inp_pad[:,0:GENERATE_LENGTH+1,:] = inp[:,0:GENERATE_LENGTH+1,:]
                tour, cost, valid = model.generate(inp_pad, GENERATE_LENGTH)
                print("----------generated output-----------")
                print('COST =',cost, ' BEST COST=', best_cost, '\n tour=',tour)

                if cost < best_cost:
                    print('-----------saving best validation model-------')
                    save_model(model,epoch,optim,SAVE_FILE_NAME)
                    best_cost = cost
        print('epoch completed=', epoch, ' -------------*****************--------------')
        avg_percent_optimal = validate_model(model, x_val, masked_output_val, opt_cost_val)
        print("validation: ----average percent optimal =", avg_percent_optimal)

if __name__ == "__main__":
    sys.exit(int(main() or 0))

