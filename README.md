# Mandalore

A two-phase platform for multiple transformers, multiple pruning methods, and multiple hardware architectures.

Basic training, inference, pruning of model is done on /model. 
  for now, Sanger and Wanda are implemented. 
  export PROJ_ROOT as current directory before any code run. 
  sh ./scripts/train_sanger_on_squad.sh for Sanger training.
  sh ./scripts/eval_sanger_on_squad.sh for for Sanger evaluation.  
	(same for Wanda)

   if there is an issue with AttributeError: module 'distutils' has no attribute 'version', try 'conda/pip install setuptools==59.5.0'
   useful post: https://sebhastian.com/python-attributeerror-module-distutils-has-no-attribute-version/

Known Issues:
   Batching causes error for modeling_bert_Wanda


Mandalore Workflow:
Phase 1] Extraction with PyTorch Hook. 
"python extraction.py --prune_method sanger/bert"
Phase 2] Running on Simulator. (requires tensors_with_layer_data.pt from Phase1)
"python simulator.py"

