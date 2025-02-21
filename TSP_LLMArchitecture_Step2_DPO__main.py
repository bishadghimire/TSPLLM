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
import torch.nn.functional as F

NUM_EPOCHS = int(20)
BATCH_SIZE = 32
GRADIENT_ACCUMULATE_EVERY = 1
LEARNING_RATE = 1e-4
VALIDATE_EVERY  = 1000
GENERATE_EVERY  = 1000
NUM_NODES = 76
GENERATE_LENGTH = NUM_NODES 
SEQ_LENGTH = 2 * NUM_NODES 
EMBEDDING_SIZE = 384 
NUM_LAYERS = 12 
NUM_HEADS = 6
LATENT_LEN = NUM_NODES 
RESUME_TRAINING = False
DO_INFERENCE_ONLY = False 
BETA = 0.1 
STEP1_TRAINED_MODEL_FILE = "checkpoint/TSPModel_12_6_384_Nodes76_eil76_STEP1.pt"
SAVE_MODEL_FILE_STEP2_DPO = "TSPModel_12_6_384_Nodes76_ARY_STEP2.pt"

TrainDataset_File = "data/TSP_TrainingData_eli76_100000.txt"
TSPLibDataset_File = "data/TSPTrainingData_EilE76_Opt.txt"
ValidationDataset_File = "data/TSPTrainingData_Test_for_eil76_10.txt"


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

def validate_model(model, val_loader):
    x, masked_output, opt_cost = next(val_loader)
    total_percent_cost = 0
    for i in range(0,x.shape[0]):
        inp_pad_val = torch.zeros((1,SEQ_LENGTH+1,x.shape[2])).cuda()
        inp_pad_val[:,0:GENERATE_LENGTH+1,:] = x[i:i+1,0:GENERATE_LENGTH+1,:]
        tour_val, cost_val, valid = model.generate(inp_pad_val, GENERATE_LENGTH)
        percent_optimal = ((cost_val - opt_cost[i])/opt_cost[i])*100
 
        total_percent_cost = total_percent_cost + ((cost_val - opt_cost[i])/opt_cost[i])*100
    return total_percent_cost/x.shape[0]


def main():
    beta = BETA
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
    
    PAR_model_ref = PerceiverARTransformer(
        dim = EMBEDDING_SIZE, 

        num_unique_tokens = NUM_NODES,
        num_layers = NUM_LAYERS, 
        heads = NUM_HEADS, 
        sequence_len = SEQ_LENGTH,
        latent_len = LATENT_LEN
     ).cuda()

    model = AutoRegressiveWrapper(PAR_model,latent_len=LATENT_LEN)
    model_ref = AutoRegressiveWrapper(PAR_model_ref,latent_len=LATENT_LEN)
    optim = configure_optimizers(model)    
    model.cuda()
    model_ref.cuda()
    
    pcount = count_parameters(model)
    print("count of parameters in the model = ", pcount/1e6, " million")
    if DO_INFERENCE_ONLY == False:  
        train_loader, val_loader, val_dataset = Utils.get_loaders_TSP(TrainDataset_File, SEQ_LENGTH, BATCH_SIZE)
    gen_loader = Utils.get_loaders_TSPLib(TSPLibDataset_File,200,1) 
    gen_loader_val = Utils.get_loaders_TSPLib(ValidationDataset_File,200,10)

    optim_RL = torch.optim.Adam(model.parameters(), lr = LEARNING_RATE)     

    best_cost = 999999
    
    if RESUME_TRAINING == False:
        checkpoint_data = torch.load(STEP1_TRAINED_MODEL_FILE)
        model.load_state_dict(checkpoint_data['state_dict'])
        optim.load_state_dict(checkpoint_data['optimizer'])
        start = 0
 
    else:  
        checkpoint_data = torch.load('checkpoint/' + SAVE_MODEL_FILE_STEP2_DPO)
        model.load_state_dict(checkpoint_data['state_dict'])
        optim.load_state_dict(checkpoint_data['optimizer'])

        for param_group in optim.param_groups:  
            param_group['lr'] = LEARNING_RATE
        start = checkpoint_data['epoch']
    checkpoint_data_ref = torch.load(STEP1_TRAINED_MODEL_FILE) 
    model_ref.load_state_dict(checkpoint_data_ref['state_dict'])
    
    if DO_INFERENCE_ONLY == True:
        avg_percent_optimal = validate_model(model, gen_loader_val)
        print("average percent optimal =", avg_percent_optimal)

    model_ref.eval()
    inp, _, _ = next(gen_loader)
        
    inp_pad_test = torch.zeros((inp.shape[0],SEQ_LENGTH+1,inp.shape[2])).cuda()
    inp_pad_test[:,0:GENERATE_LENGTH+1,:] = inp[:,0:GENERATE_LENGTH+1,:]

    model.eval()
    tour, cost, valid = model.generate(inp_pad_test, GENERATE_LENGTH)
    best_cost = cost
    print("----------generated output-----------")
    print('COST =',cost, ' tour=',tour)
    print("----------end generated output-----------")
    Utils.plotTSP(tour, inp,1)
    if DO_INFERENCE_ONLY == True:
        return
    
 
    x_val, masked_output_val, opt_cost_val = next(gen_loader_val) 
    for epoch in range(start, NUM_EPOCHS): 
        for i, data in enumerate(train_loader):    
            x, masked_output = data
            model_ref.eval()
            model.eval()
            optim.zero_grad()
 
            actual_tour_model = torch.zeros((x.shape[0], GENERATE_LENGTH))
            actual_tour_modelref = torch.zeros((x.shape[0], GENERATE_LENGTH))
            for j in range(0,x.shape[0]):
                inp_pad = torch.zeros((1,SEQ_LENGTH+1,x.shape[2])).cuda()
                inp_pad[:,0:GENERATE_LENGTH+1,:] = x[j:j+1,0:GENERATE_LENGTH+1,:]
                tour1, cost1, valid1 = model.generate(inp_pad, GENERATE_LENGTH, with_topk=False)
                tour2, cost2, valid2 = model_ref.generate(inp_pad, GENERATE_LENGTH, with_topk=False)
                actual_tour_model[j] = torch.tensor(tour1[1:])
                actual_tour_modelref[j] = torch.tensor(tour2[1:])
 
            model.train()
            pi_yl_logps, pi_yw_logps = model.generate_for_DPO(x, GENERATE_LENGTH,actual_tour_model)
            ref_yl_logps, ref_yw_logps = model_ref.generate_for_DPO(x, GENERATE_LENGTH,actual_tour_modelref)

            pi_logratios = pi_yw_logps - pi_yl_logps
            ref_logratios = ref_yw_logps - ref_yl_logps

            loss = (-F.logsigmoid(beta * (pi_logratios - ref_logratios))).mean()
            loss.backward()
            optim.step()
            optim.zero_grad()
            print(f'training loss: {loss.item()} -- iteration = {i}')
        
            if i % 10 == 0:
                model.eval()
            
                tour, cost, valid = model.generate(inp_pad_test, GENERATE_LENGTH)
                print("----------generated output-----------")
                print('COST =',cost, ' tour=',tour, ' best_cost=', best_cost)
                if cost < best_cost:
                    save_model(model, epoch, optim, SAVE_MODEL_FILE_STEP2_DPO)
                    best_cost = cost

            if i == 1000:
                optim.param_groups[0]["lr"] = 3e-4
                beta = beta/2
            if i == 1500:
                optim.param_groups[0]["lr"] = 1e-5
                beta = beta/2
        

if __name__ == "__main__":
    sys.exit(int(main() or 0))


