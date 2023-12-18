import torch 
import torch.nn as nn
from model import *
import numpy as np
import argparse
import os

PROJ_ROOT = os.environ.get('PROJ_ROOT')
CONFIG_ROOT=PROJ_ROOT+"/model/configs" #do remember to export PROJ_ROOT environment variable! 

from transformers import (
    AutoConfig,
    AutoModelForQuestionAnswering,
    AutoTokenizer,
    DataCollatorWithPadding,
    EvalPrediction,
    HfArgumentParser,
    PreTrainedTokenizerFast,
    TrainingArguments,
    default_data_collator,
    set_seed,
)

parser = argparse.ArgumentParser(description="Process prune method.")
    
# Adding the --prune_method argument as a string
parser.add_argument('--prune_method', type=str, required=True,
                        help='Specify the method to prune. wanda or sanger')
# Parse arguments
args = parser.parse_args()

import model.quant_utils 
if args.prune_method == 'wanda':
    from model.modeling_bert_Wanda import BertForQuestionAnswering
    target_module_list = [nn.Linear, model.modeling_bert_Wanda.CustomMatmul] #quant_utils.QuantizedMatMul, 
elif args.prune_method == 'sanger':
    from model.modeling_bert_Sanger import BertForQuestionAnswering
    target_module_list = [nn.Linear, model.modeling_bert_Sanger.CustomMatmul] #quant_utils.QuantizedMatMul,  
else:
    raise Exception("Prune Method not supported")

config = AutoConfig.from_pretrained(CONFIG_ROOT+"/bert_base_sanger_2e-3.json")
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
model_name = 'bert-base-uncased'
#line 621 of run_squad.py
#from modeling_bert import BertForQuestionAnswering
#from modeling_bert_Wanda import BertForQuestionAnswering
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('verify:', device)
model_ = BertForQuestionAnswering.from_pretrained(model_name, from_tf=False,config=config).to(device)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
state_dict = torch.load("model/outputs/squad/sparse-bert-base-uncased/pytorch_model.bin", map_location=device)
#print(state_dict)
model_.load_state_dict(state_dict)
#model = model.to(device)

#Stat_Collector from https://github.com/SamsungLabs/Sparse-Multi-DNN-Scheduling/blob/main/dataset_sparsity/data_util.py
global_tensor_cache = [] #global array can be easily added.
global_layerinfo_cache = []

def trim_matrix(matrix):
  if matrix.dim() > 2:
    matrix = matrix.squeeze(0)
  return matrix

class Stat_Collector:
  def __init__(self, m):
    self.handle = m.register_forward_hook(self.hook_fn)
    #self.sparsity = 0.0
  def hook_fn(self, m, inp, outp): #attached to that model with Stat_Collector initialization
    self.out_features = outp.clone()
    self.m = m
    o_shape = self.out_features.shape
    if isinstance(self.m, nn.Linear):
      self.in_features = inp
      i_shape = self.in_features[0].shape #take tensor out of the tuple
      if (self.out_features.dtype is torch.float32):
        weight = self.m.weight
      else:
        weight = self.m.weight()
      w_shape = weight.shape
      print('module is:', m)
      print("i_shape:", i_shape)#, "input feature: ", self.in_features)
      print("w_shape:", w_shape)
      print("o_shape:", o_shape)
      export1 = trim_matrix(self.in_features[0]) 
      export2 = trim_matrix(self.m.weight)
      export3 = trim_matrix(self.out_features)
      global_tensor_cache.append((export1, export2.t(), export3)) #transpose export2 so it can be tranposed in the simulator
      #global_tensor_cache.append((self.in_features[0], self.m.weight, self.out_features))
      #global_layerinfo_cache.append((str(m), i_shape, w_shape, o_shape))
      global_layerinfo_cache.append((str(m), export1.shape, export2.shape, export3.shape))
      #torch.save(tensors, "tensors_tuple.pt")
      #print(f"Layer Name: {name}, Layer Type: {type(module)}")
    #deprecated due to unnecessary prune logic
    #elif (isinstance(self.m, quant_utils.QuantizedMatMul)):
    #  print('module is:', m)
    #  self.in_features_1 = inp[0]
    #  self.in_features_2 = inp[1]
    #  x_shape = self.in_features_1.shape
    #  y_shape = self.in_features_2.shape
    #  print("x_shape:", x_shape)
    #  print("y_shape:", y_shape)
    #  print("o_shape:", o_shape)
    #  global_layerinfo_cache.append((str(m), x_shape, y_shape, o_shape))
    #  global_tensor_cache.append((self.in_features_1, self.in_features_2, self.out_features))
    elif (args.prune_method == 'wanda'):
        if (isinstance(self.m, model.modeling_bert_Wanda.CustomMatmul)):
            print('module is:', m)
            self.in_features_1 = inp[0]
            self.in_features_2 = inp[1]
            x_shape = self.in_features_1.shape
            y_shape = self.in_features_2.shape
            print("x_shape:", x_shape)
            print("y_shape:", y_shape)
            print("o_shape:", o_shape)
            #special logic for exporting this batch matmul.(where each channel is a head) 
            export1 = trim_matrix(self.in_features_1)
            export2 = trim_matrix(self.in_features_2)
            export3 = trim_matrix(self.out_features)
            global_tensor_cache.append((export1, export2, export3))
            #global_layerinfo_cache.append((str(m), x_shape, y_shape, o_shape))
            global_layerinfo_cache.append((str(m), export1.shape, export2.shape, export3.shape))
    elif (args.prune_method == 'sanger'):
        if (isinstance(self.m, model.modeling_bert_Sanger.CustomMatmul)):
            print('module is:', m)
            self.in_features_1 = inp[0]
            self.in_features_2 = inp[1]
            x_shape = self.in_features_1.shape
            y_shape = self.in_features_2.shape
            print("x_shape:", x_shape)
            print("y_shape:", y_shape)
            print("o_shape:", o_shape)
            #special logic for exporting this batch matmul.(where each channel is a head) 
            export1 = trim_matrix(self.in_features_1)
            export2 = trim_matrix(self.in_features_2)
            export3 = trim_matrix(self.out_features)
            global_tensor_cache.append((export1, export2, export3))
            #global_layerinfo_cache.append((str(m), x_shape, y_shape, o_shape))
            global_layerinfo_cache.append((str(m), export1.shape, export2.shape, export3.shape))
    else:
      raise NameError("Hook Layer not supported")
   
  def remove(self):
    self.handle.remove()

  # Insert hook of every "target_module"
  # Return the inserted model and intermediate result 
  def insert_hook(model, target_module_list):
    extern_output = []
    intern_outputs = []
    for name, layer in dict(model.named_modules()).items(): # donghyeon: so this does not seem to go in recursive
      # print (layer.__class__)
      for target_module in target_module_list:
        if isinstance(layer, target_module):
          print("Collect: ", name, layer)
          extern_output.append((name, layer))
          intern_outputs.append(Stat_Collector(layer))
    return model, intern_outputs, extern_output
  
#target_module_list = [nn.Linear, modeling_bert_Wanda.CustomMatmul] #quant_utils.QuantizedMatMul,    => Done up there. 

model_, _, layer_log = Stat_Collector.insert_hook(model_, target_module_list) #inserting hooks

with open("BERT_layers.txt", "w") as file:
    for _, layer in model_.named_modules():
        file.write(str(layer))
        file.write("\n-------------------\n")

with open("BERT_target_layers.txt", "w") as file:
    for name, layer in dict(model_.named_modules()).items(): # donghyeon: so this does not seem to go in recursive
      for target_module in target_module_list:
        if isinstance(layer, target_module):
          file.write(str(layer)+"\n")

question = "How many programming languages does BLOOM support?"
context = "The number of programming languages supported varies from language models. However, BLOOM has 176 billion parameters and can generate text in 46 languages natural languages and 13 programming languages. The performance in languages other than English can be less reliable, especially for languages that are less represented in the training data. This includes many regional or less commonly spoken languages. The quality of understanding and generating text in these languages can be lower, and the model may struggle with complex tasks or nuanced conversations in those languages. On the other hand, ChatGPT has the ability to understand and generate text in several languages."

inputs = tokenizer(question, context, return_tensors="pt")
with torch.no_grad():
    outputs = model_(**inputs)

#An integrated layer_info - matrix operands 
tensors_with_layer_data = []
if (len(global_tensor_cache) == len(global_layerinfo_cache)):
    print("log-matrix length match with length: ", len(global_layerinfo_cache))
    for i in range(0, len(global_tensor_cache)):
        tensors_with_layer_data.append((global_layerinfo_cache[i], global_tensor_cache[i]))
else:
    print("length mismatch: ", len(global_layerinfo_cache), len(global_tensor_cache))
torch.save(tensors_with_layer_data, "tensors_with_layer_data.pt")

with open("BERT_layerinfo_runtime.txt", "w") as file:
    for i in global_layerinfo_cache:
        file.write(str(i) + "\n")