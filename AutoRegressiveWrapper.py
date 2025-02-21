from pickle import FALSE
import torch
from torch import nn
import torch.nn.functional as F
import math
import itertools

def top_k(logits, thres = 0.9):
    k = int((1 - thres) * logits.shape[-1])
    val, ind = torch.topk(logits, k)
    probs = torch.full_like(logits, float('-inf')) 
    probs.scatter_(1, ind, val) 
    return probs                

class AutoRegressiveWrapper(nn.Module):
    def __init__(self, net, latent_len, pad_value = 0):
        super().__init__()
        self.pad_value = pad_value
        self.model = net
        self.latent_len = latent_len
        self.max_seq_len = net.sequence_len

    def generate(self, x, generate_len, with_topk=False):
        opt_seq =[]

        valid_tour = True
        prev_out = x[:,0:-1,:]  
        masked_output = torch.ones((x.shape[0],generate_len,generate_len)).cuda()
        predicted_node_seq = []
        predicted_node_seq.append(0)  
        pos = generate_len+1
        mask_out = torch.ones((1,generate_len)).cuda()
        mask_out[:,0] = 0
        predicted_probs = torch.zeros((generate_len-1))
        target_probs = torch.zeros((generate_len-1)) 
        for i in range(0,generate_len-1):    

            logits, value_out = self.model(prev_out, masked_output)
            if with_topk == True:

                filter_thres = 0.94
                temperature = 1.0
                minv = torch.min(logits[:,i,:])
                if minv < 0:
                    minv = -1 *minv
                else:
                    minv = 0
                
                filtered_logits = top_k((logits[:,i,:]+minv)*mask_out, thres = filter_thres)

                probs = F.softmax(filtered_logits / temperature, dim=-1)
                predicted_index = torch.multinomial(probs, 1).item() 
            if with_topk == False:

                probs_wout_masking = F.softmax(logits[0,i,:], dim=-1)
                probs = F.softmax(logits[0,i,:], dim=-1)*mask_out  

                predicted_index = torch.argmax(probs, dim=1).item()

            predicted_prob = probs[0,predicted_index]

            with torch.no_grad():
                predicted_probs[i] = torch.log(predicted_prob)
            mask_out[:,predicted_index] = 0

            predicted_node_seq.append(predicted_index)
            predicted_node = x[:,predicted_index,:].clone()

            prev_out[:,pos,:] = predicted_node  
            pos = pos + 1
        predicted_node_seq.append(0)
 
        cost = self.determine_tour_cost(x, generate_len, predicted_node_seq )
        

        valid_nums = torch.arange(0,generate_len)
        for i in range(0,generate_len):
            for k in range(0,generate_len):
                found = False
                if predicted_node_seq[k] == i:
                    found = True
                    break
            if found == False:
                print('*****Node ', i, not found)
                valid_tour = False
        
        return predicted_node_seq, cost, valid_tour 

    def determine_tour_cost(self, x, generate_len, predicted_node_seq):
        cost = 0
        curr_node_x = x[:,generate_len,1].item()
        curr_node_y = x[:,generate_len,2].item()
        start_node_x = curr_node_x
        start_node_y = curr_node_y
        for i in range(0,generate_len):
            next_node_x = x[:,predicted_node_seq[i],1].item()
            next_node_y = x[:,predicted_node_seq[i],2].item()
            dist = math.sqrt((curr_node_x - next_node_x)**2 + (curr_node_y - next_node_y)**2)
            cost = cost + dist
            curr_node_x = next_node_x
            curr_node_y = next_node_y
        dist = math.sqrt((curr_node_x - start_node_x)**2 + (curr_node_y - start_node_y)**2)
        cost = cost + dist
        return cost
    
    def forward(self, x, masked_output):
        xi = x[:, :-1] 
        xo = x[:, 1:,0]
        xo = xo[:,self.latent_len:]
        out, value_out = self.model(xi, masked_output)
        out = out[:,-(self.latent_len):,:] * masked_output
        logits_reorg = out.reshape(-1, out.size(-1))
        targets_reorg = xo.reshape(-1).long()
        loss = F.cross_entropy(logits_reorg, targets_reorg)
        return loss

    def generate_for_RL(self, x, generate_len):
        self.model.train()
        prev_out = x[:,0:-1,:]  
        masked_output = torch.ones((x.shape[0],generate_len,generate_len)).cuda()
        predicted_node_seq = []
        predicted_node_seq.append(0)  
        pos = generate_len+1
        mask_out = torch.ones((1,generate_len)).cuda()
        mask_out[:,0] = 0
        for i in range(0,generate_len-1):   
            logits, value_out = self.model(prev_out, masked_output)
            probs = F.softmax(logits[0,i,:], dim=-1)*mask_out
            predicted_index = torch.argmax(probs, dim=-1)
            mask_out[:,predicted_index.item()] = 0
            predicted_node_seq.append(predicted_index.item())
            predicted_node = x[:,predicted_index.item(),:].clone()
            prev_out[:,pos,:] = predicted_node  
            pos = pos + 1
        predicted_node_seq.append(0)

        cost = torch.tensor(0.0,requires_grad=True)

        curr_node_x = x[:,generate_len,1]
        curr_node_y = x[:,generate_len,2]
        start_node_x = curr_node_x
        start_node_y = curr_node_y
        for i in range(0,generate_len):
            next_node_x = x[:,predicted_node_seq[i],1]
            next_node_y = x[:,predicted_node_seq[i],2]
            dist = math.sqrt((curr_node_x - next_node_x)**2 + (curr_node_y - next_node_y)**2)
            cost = cost + dist
            curr_node_x = next_node_x
            curr_node_y = next_node_y
        dist = math.sqrt((curr_node_x - start_node_x)**2 + (curr_node_y - start_node_y)**2)
        cost = cost + dist
        return cost
    
    def best_tour_permutek(self, x, generate_len, predicted_node_seq, k=5):

        cost = self.determine_tour_cost(x, generate_len, predicted_node_seq)
        best_permuted_cost = cost
        tour = predicted_node_seq
        for i in range(1, len(tour)-k):
            partial_tour = tour[i:i+k]
            permute_partial_tour = list(itertools.permutations(partial_tour))

            for j in range(0,len(permute_partial_tour)):
                full_tour_with_permutation = tour[0:i]+list(permute_partial_tour[j]) + tour[i+k:]

                cost_permuted_tour = self.determine_tour_cost(x, generate_len, full_tour_with_permutation)
                if cost_permuted_tour < best_permuted_cost:
                    print('better tour found cost=', cost_permuted_tour)
                    tour = full_tour_with_permutation
                    best_permuted_cost = cost_permuted_tour
        return tour, best_permuted_cost
    
    def best_tour_permutek_skip(self, x, generate_len, predicted_node_seq, k=5):

        cost = self.determine_tour_cost(x, generate_len, predicted_node_seq)
        best_permuted_cost = cost
        tour = predicted_node_seq
        m = k+k+k+k
        for i in range(1, len(tour)-(m+5)):
            partial_tour = tour[i:i+3] + tour[i+m:i+m+2]
            permute_partial_tour = list(itertools.permutations(partial_tour))

            for j in range(0,len(permute_partial_tour)):
                aa = list(permute_partial_tour[j])[0:3]
                bb = list(permute_partial_tour[j])[3:k]
                cc =  tour[i+m+2:]
                full_tour_with_permutation = tour[0:i]+list(permute_partial_tour[j])[0:3] + tour[i+3:i+m] + list(permute_partial_tour[j])[3:k] + tour[i+m+2:]
                dd = len(full_tour_with_permutation)

                cost_permuted_tour = self.determine_tour_cost(x, generate_len, full_tour_with_permutation)
                if cost_permuted_tour < best_permuted_cost:
                    print('better tour found cost 2=', cost_permuted_tour)
                    tour = full_tour_with_permutation
                    best_permuted_cost = cost_permuted_tour
        return tour, best_permuted_cost
    
    def generate_for_DPO(self, x, generate_len, tour):

        optimal_tour = x[:,generate_len+1:generate_len+generate_len+1,0] 
        prev_out = x[:,0:-1,:]  
        masked_output = torch.ones((x.shape[0],generate_len,generate_len)).cuda()
        logits, value_out = self.model(prev_out, masked_output) 

        probs_wout_masking = F.log_softmax(logits, dim=-1)

        optimal_tour = optimal_tour.view(-1,optimal_tour.shape[1],1).long().cuda()
        actual_tour = tour.view(-1,optimal_tour.shape[1],1).long().cuda()
        optimal_tour_probs = torch.gather(probs_wout_masking, dim=-1, index=optimal_tour)
        actual_tour_probs = torch.gather(probs_wout_masking, dim=-1, index=actual_tour)
        predicted_logsum = actual_tour_probs.squeeze(dim=-1)
        target_logsum = optimal_tour_probs.squeeze(dim=-1)
        return predicted_logsum, target_logsum 