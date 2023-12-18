import torch 
import numpy as np
import argparse
import os

#prompt
#question = "How many programming languages does BLOOM support?"
#context = "The number of programming languages supported varies from language models. However, BLOOM has 176 billion parameters and can generate text in 46 languages natural languages and 13 programming languages. The performance in languages other than English can be less reliable, especially for languages that are less represented in the training data. This includes many regional or less commonly spoken languages. The quality of understanding and generating text in these languages can be lower, and the model may struggle with complex tasks or nuanced conversations in those languages. On the other hand, ChatGPT has the ability to understand and generate text in several languages."
context = "Korea university programming club used to have 30 members and 5 project teams."
question = "What was the name of the programming club?"
##

PROJ_ROOT = os.environ.get('PROJ_ROOT')
CONFIG_ROOT=PROJ_ROOT+"/model/configs" #do remember to export PROJ_ROOT environment variable! 

#call saved model
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

config = AutoConfig.from_pretrained(CONFIG_ROOT+"/bert_base_sanger_2e-3.json")
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
model_name = 'bert-base-uncased'

if args.prune_method == 'wanda':
    from modeling_bert_Wanda import BertForQuestionAnswering
elif args.prune_method == 'sanger':
    from modeling_bert_Sanger import BertForQuestionAnswering
else:
    raise Exception("Prune Method not supported")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('verify:', device)
model = BertForQuestionAnswering.from_pretrained(model_name, from_tf=False,config=config).to(device)

state_dict = torch.load("outputs/squad/sparse-bert-base-uncased/pytorch_model.bin", map_location=device)
model.load_state_dict(state_dict)


inputs = tokenizer(question, context, return_tensors="pt")
with torch.no_grad():
    outputs = model(**inputs) 


#print output
answer_start_index = outputs.start_logits.argmax()
answer_end_index = outputs.end_logits.argmax()
predict_answer_tokens = inputs.input_ids[0, answer_start_index : answer_end_index + 1]
print(tokenizer.decode(predict_answer_tokens))