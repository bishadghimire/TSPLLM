# One-shot Autoregressive Generation of Combinatorial Optimization Solutions based on the Large Language Model Architecture and Learning Algorithms

Abstract: Large Language Models (LLMs) have immensely advanced the field of Artificial Intelligence (AI) with recent models being able to perform chain of thought reasoning and solve complex mathematical problems ranging from theorem proving to ones involving advanced calculus. The success of the LLMs derives from a combination of the Transformer architecture with its attention mechanism, the autoregressive training methodology with masked attention, and the alignment finetuning via reinforcement learning algorithms. In this research, we attempt to explore a possible solution to the fundamental NP-hard problem of combinatorial optimization, in particular the Travelling Salesman Problem (TSP), by following the LLM approach in terms of architecture and the training algorithms. Similar to the LLM design, which is trained in an autoregressive manner to predict the next token, our model is trained to predict the next-node in a TSP graph. After the model is trained on random TSP graphs with known near-optimal solutions, we fine tune the model using Direct Preference Optimization (DPO). The tour generation in a trained model is autoregressive one step generation with no need for iterative refinement. Our results are very promising and indicate that for TSP graphs, for up to 100 nodes, a reasonably small amount of training data yields solutions within a few percent of the optimal. This optimization improves if more data is used to train the model.

Keywords: Large Language Models; Transformer; Reinforcement Learning; Combinatorial Optimization; Travelling Salesman Problem

PrePrint available @ https://www.preprints.org/manuscript/202502.1797/v1

**Instructions**

Run the "TSP_LLMArchitecture_Step1_CE_main.py" for the specified model size (embedding dimensions, number of layers, number of heads) and the training data files. These settings are specified at the top of "TSP_LLMArchitecture_Step1_CE_main.py". For example, to train the model for 29 nodes problem and test it on Bays29 TSPLIB benchmark, the settings are as follows:

NUM_EPOCHS = int(25) __
BATCH_SIZE = 16 __
GRADIENT_ACCUMULATE_EVERY = 1 <br/>
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

To obtain training data for different size graphs, you can contact the authors.



TSP_LLMArchitecture_Step2_DPO__main.py
