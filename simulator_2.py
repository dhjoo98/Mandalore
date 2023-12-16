import torch 
import time 
#from stat_collection import *

#global_counter = 0

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Mandalore:
    def __init__(self, num_PEs = 64):
        self.num_PEs = num_PEs
        self.tiler = Tiler()
        self.multiplier = MultiplierArray(num_PEs) #which also logs #ofcycles and #ofidlePEs
        self.adder = AdderTree()
        self.global_counter = 0
        #self.output = 

    def process(self, packed_network):
        total_layers = len(packed_network)
        for metadata, (x, y, z) in packed_network:
            if metadata[0].startswith("CustomMatmul") and metadata[1] == torch.Size([12, 128, 64]): #QK 
                print("starting QK Calculation")
                for h in range(0,metadata[1][0]):
                    x_ = x[h].squeeze()
                    y_ = y[h].squeeze()
                    #print("shapes: ", x_.shape, y_.shape)
                    tiled_Q = self.tiler.process(x_,y_)
                    tiled_Q_post_mul = self.multiplier.process(tiled_Q)
                    del tiled_Q
                    output_single_elements = self.adder.process(tiled_Q_post_mul)
                    del tiled_Q_post_mul
                    row = x_.shape[0]
                    col = y_.shape[1]
                    output_mat = torch.zeros(row, col)
                    for i in range(0,len(output_single_elements)):
                        output_mat[output_single_elements[i][1], output_single_elements[i][2]] += output_single_elements[i][0]
                    del output_single_elements    
                self.global_counter += 1
                print(f"------{self.global_counter}/{total_layers} completed computation!")
            elif metadata[0].startswith("CustomMatmul") and metadata[1] == torch.Size([12, 128, 128]): #SV
                
                #this is analytical, but with the number of zeros of S matrix, I can discount operations of QK of that coordinate
                ## = 64 x number of tiles x number of zeros (does not affect the number of idle PE bubble )

                print("starting SV Calculation")
                for h in range(0,metadata[1][0]):
                    x_ = x[h].squeeze()
                    y_ = y[h].squeeze()
                    #print("hmm  shapes: ", x_.shape, y_.shape)
                    tiled_Q = self.tiler.process(x_,y_)
                    tiled_Q_post_mul = self.multiplier.process(tiled_Q)
                    del tiled_Q
                    output_single_elements = self.adder.process(tiled_Q_post_mul)
                    del tiled_Q_post_mul
                    row = x_.shape[0]
                    col = y_.shape[1]
                    output_mat = torch.zeros(row, col)
                    for i in range(0,len(output_single_elements)):
                        output_mat[output_single_elements[i][1], output_single_elements[i][2]] += output_single_elements[i][0] 
                    del output_single_elements
                self.global_counter += 1
                print(f"------{self.global_counter}/{total_layers} completed computation!")
            else:
                print("starting Linear!")
                tiled_Q = self.tiler.process(x,y)
                tiled_Q_post_mul = self.multiplier.process(tiled_Q)
                del tiled_Q
                output_single_elements = self.adder.process(tiled_Q_post_mul)
                del tiled_Q_post_mul
                row = x.shape[0]
                col = y.shape[1]
                output_mat = torch.zeros(row, col)
                for i in range(0,len(output_single_elements)):
                    output_mat[output_single_elements[i][1], output_single_elements[i][2]] += output_single_elements[i][0]
                del output_single_elements
                self.global_counter += 1
                print(f"------{self.global_counter}/{total_layers} completed computation!")
        print("Matmul Simulation complete!\n")
        print("Number of Cycles: ", self.multiplier.cycle_taken)
        print("Number of idle PE: ", self.multiplier.idle_PE_bubble)
        print("PE Utilization:", self.multiplier.cycle_taken * self.num_PEs / self.multiplier.idle_PE_bubble )    
        #no returns necessary 

class Tiler:
    def process(self, matrix1, matrix2):
        #perform matrix tiling
        PE_array_width = 64
        block_M = matrix1.shape[0]
        block_N = int(matrix1.shape[1] / PE_array_width) 
        block_K = matrix2.shape[1] 
        tiled_Q = []
        #print(matrix1.shape, matrix2.shape, block_M, block_N, block_K)
        transposed_B = matrix2.t()
        for i in range(0, block_M):
            for j in range(0, block_K):
                    for k in range(0, block_N):
                        tiled_Q.append([matrix1[i][k*PE_array_width:(k+1)*PE_array_width],transposed_B[j][k*PE_array_width:(k+1)*PE_array_width], i, j])
        print("the length of the queue:", len(tiled_Q))
        return tiled_Q

#core component
#use something like a filter, to detect 1) zero elements, and 2) zero row! 
class MultiplierArray:
    def __init__(self, length=64):
        self.length = length
        self.cycle_taken=0
        self.idle_PE_bubble=0 #what determines PE_utilization the most: 

    def process(self, tiled_Q):
        # Simple element-wise multiplication for demonstration
        tiled_Q_post_mul = []
        self.cycle_taken = len(tiled_Q)
        for i in range(0, len(tiled_Q)):
            workload = tiled_Q[i]
            result = workload[0]*workload[1]
            tiled_Q_post_mul.append([result, workload[2], workload[3]])
            self.idle_PE_bubble += torch.eq(result,0).sum().item()
        return tiled_Q_post_mul

class AdderTree:
    def process(self, tiled_Q_post_mul):
        output_single_elements = []
        for i in range(0,len(tiled_Q_post_mul)):
            output_single_elements.append([torch.sum(tiled_Q_post_mul[i][0]).item(), tiled_Q_post_mul[i][1], tiled_Q_post_mul[i][2]])
        return output_single_elements
    

loaded_tensors = torch.load('tensors_with_layer_data.pt')

start_time = time.time()
mandalore = Mandalore()
mandalore.process(loaded_tensors)
end_time = time.time()
elapsed_time = end_time - start_time
minutes, seconds = divmod(elapsed_time, 60)
print(f"Elapsed time: {int(minutes)} minutes and {seconds:.2f} seconds")

