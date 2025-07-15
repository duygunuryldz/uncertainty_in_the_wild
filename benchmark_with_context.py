import os 
import sys
sys.path.append('../TruthTorchLM/src/')
import TruthTorchLM as ttlm
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast, PreTrainedModel
import json
import ast
from typing import Union
os.environ['SERPER_API_KEY'] = 'YOUR_KEY'
os.environ["OPENAI_API_KEY"] = 'YOUR_KEY'
from datasets import load_dataset
import wandb
import time
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer, DebertaForSequenceClassification, DebertaTokenizer
import pickle
import argparse
from TruthTorchLM.templates import DEFAULT_SYSTEM_PROMPT, DEFAULT_USER_PROMPT, SELF_DETECTION_QUESTION_PROMPT, SELF_DETECTION_SYSTEM_PROMPT, ENTAILMENT_PROMPT, DEFAULT_SYSTEM_BENCHMARK_PROMPT 
from TruthTorchLM.templates import PTRUE_SYSTEM_PROMPT, PTRUE_USER_PROMPT, PTRUE_MODEL_OUTPUT
import pandas as pd
import random


args = argparse.ArgumentParser()
args.add_argument("--model_name", type=str, default='meta-llama/Llama-2-7b-chat-hf')
args.add_argument("--device", type=int, default = 0)
args.add_argument("--dataset_name", type=str, default='trivia_qa')
args.add_argument("--dataset_size", type=int, default=1000)
args.add_argument('--original_dataset_calibration_size', type=int, default=100)

args.add_argument("--number_of_generations", type=int, default=5)
args.add_argument("--seed", type=int, default=0)
args.add_argument('--model_judge', type=str, default='gpt-4o-mini')
args.add_argument('--context_dataset', type=str, default='natural_qa')
args.add_argument('--num_context_examples', type=int, default=5)
args.add_argument('--auto_device', type=bool, default=False)
args.add_argument('--save_name', type=str, default='benchmark_with_context')

args = args.parse_args()

number_of_generations = args.number_of_generations
seed = args.seed
model_name = args.model_name
device = args.device
dataset_name = args.dataset_name
dataset_size= args.dataset_size
context_dataset = args.context_dataset
num_context_example = args.num_context_examples
original_dataset_calibration_size = args.original_dataset_calibration_size

max_new_tokens = 128

device = torch.device("cuda:" + str(device) if torch.cuda.is_available() else "cpu")
wandb_name = f'{args.save_name}_{dataset_name}_{model_name}_{dataset_size}_{number_of_generations}_{seed}_{context_dataset}_{num_context_example}_{original_dataset_calibration_size}'

wandb_run = wandb.init(
      # Set the project where this run will be logged
      project="uncertainty_in_the_wild_benchmark_with_context_new_gpt_fixed",
      # We pass a run name (otherwise it’ll be randomly assigned, like sunshine-lollypop-10)
      name= wandb_name,
      config={"seed": seed}
      )

if 'gpt' in model_name:
    model = model_name
    tokenizer = None
    pad_token_id = None
else:
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    pad_token_id = tokenizer.pad_token_id
    if pad_token_id == None:
        pad_token_id = model.config.eos_token_id

print(pad_token_id)

#create contexts from the given dataset
#first pick questions from the dataset
dataset = ttlm.utils.dataset_utils.get_dataset(context_dataset, split='train', seed=seed)
#pick random questions
random.seed(seed)
questions = random.sample(dataset, num_context_example)

prompt_template = 'You are a helpful assistant. Answer the following question in a single brief but complete sentence. Question: {question_context} Answer:'
chat = []
#run model on the questions and get the answers
for question in questions:
    q = question['question']
    chat.append({"role": "user", "content":prompt_template.format(question_context = q)})
    print(chat)
    if type(model) != str:
        truth_dict = ttlm.generate_with_truth_value(model=model, messages = chat, truth_methods = [], tokenizer = tokenizer, generation_seed=seed, batch_generation=True, do_sample = True, max_new_tokens = max_new_tokens, pad_token_id = pad_token_id)
    else:
        truth_dict = ttlm.generate_with_truth_value(model=model, messages = chat, truth_methods = [], tokenizer = tokenizer, batch_generation=True)
    generated_text = truth_dict['generated_text']
    chat.append({"role": "assistant", "content": generated_text})

torch.cuda.empty_cache()

model_for_entailment = DebertaForSequenceClassification.from_pretrained('microsoft/deberta-large-mnli').to(device)
tokenizer_for_entailment = DebertaTokenizer.from_pretrained('microsoft/deberta-large-mnli')

semantic_entropy = ttlm.truth_methods.SemanticEntropy(ttlm.scoring_methods.LengthNormalizedScoring(), number_of_generations=number_of_generations, model_for_entailment=model_for_entailment, tokenizer_for_entailment=tokenizer_for_entailment)
confidence = ttlm.truth_methods.Confidence(ttlm.scoring_methods.LengthNormalizedScoring())
entropy = ttlm.truth_methods.Entropy(ttlm.scoring_methods.LengthNormalizedScoring(), number_of_generations=number_of_generations)
p_true = ttlm.truth_methods.PTrue(number_of_ideas=number_of_generations)
matrix_degree_uncertainty = ttlm.truth_methods.MatrixDegreeUncertainty(number_of_generations=number_of_generations, model_for_entailment=model_for_entailment, tokenizer_for_entailment=tokenizer_for_entailment)
eccentricity_uncertainty = ttlm.truth_methods.EccentricityUncertainty(number_of_generations=number_of_generations, model_for_entailment=model_for_entailment, tokenizer_for_entailment=tokenizer_for_entailment)
sum_eigen_uncertainty = ttlm.truth_methods.SumEigenUncertainty(number_of_generations=number_of_generations, model_for_entailment=model_for_entailment, tokenizer_for_entailment=tokenizer_for_entailment)
sentSAR = ttlm.truth_methods.SentSAR(number_of_generations=number_of_generations, similarity_model_device=device)
inside = ttlm.truth_methods.Inside(number_of_generations=number_of_generations)
self_detection = ttlm.truth_methods.SelfDetection(number_of_questions = number_of_generations, method_for_similarity='semantic', model_for_entailment=model_for_entailment, tokenizer_for_entailment=tokenizer_for_entailment, pad_token_id = pad_token_id, system_prompt = None, user_prompt = prompt_template)
kernel_entropy = ttlm.truth_methods.KernelLanguageEntropy(number_of_generations=number_of_generations, model_for_entailment=model_for_entailment, tokenizer_for_entailment=tokenizer_for_entailment)
lars = ttlm.truth_methods.LARS(device = device)
eccentricity_confidence = ttlm.truth_methods.EccentricityConfidence(number_of_generations=number_of_generations, model_for_entailment=model_for_entailment, tokenizer_for_entailment=tokenizer_for_entailment)
matrix_degree_confidence = ttlm.truth_methods.MatrixDegreeConfidence(number_of_generations=number_of_generations, model_for_entailment=model_for_entailment, tokenizer_for_entailment=tokenizer_for_entailment)
attention_score = ttlm.truth_methods.AttentionScore(layer_index=23)
mars = ttlm.truth_methods.MARS(device=device, mars_temperature = 0.1)
verbalized_confidence = ttlm.truth_methods.VerbalizedConfidence(pad_token_id = pad_token_id)
sar = ttlm.truth_methods.SAR(number_of_generations=number_of_generations, similarity_model_device=device)

#meta-llama_Meta-Llama-3-8B-Instruct_trivia_qa_gsm8k_[8000, 5000]_saplma_output_dict.pkl
#load pickle
with open('meta-llama_Meta-Llama-3-8B-Instruct_trivia_qa_gsm8k_[8000, 5000]_saplma_output_dict.pkl', 'rb') as f:
    saplma_dict = pickle.load(f)

saplma_model = saplma_dict['model']
saplma = ttlm.truth_methods.SAPLMA(saplma_model=saplma_model, layer_index = -1)

if type(model) != str:
    truth_methods = [semantic_entropy, confidence, entropy, matrix_degree_uncertainty, eccentricity_uncertainty,  sum_eigen_uncertainty, sentSAR, p_true, self_detection, kernel_entropy, lars, inside, mars, attention_score, matrix_degree_confidence, eccentricity_confidence, saplma,verbalized_confidence, sar]#
else:
    truth_methods = [semantic_entropy, confidence, entropy, matrix_degree_uncertainty, eccentricity_uncertainty, sum_eigen_uncertainty, sentSAR, self_detection, kernel_entropy, lars, mars, matrix_degree_confidence, eccentricity_confidence, p_true, verbalized_confidence, sar]


eval_metrics = ['auroc', 'auprc', 'auarc','accuracy', 'f1', 'precision', 'recall', 'prr']
model_judge = ttlm.evaluators.ModelJudge(args.model_judge)


if 'greedy' in args.save_name:
    do_sample = False
else:
    do_sample = True

# if type(model) != str:
#     #first calibrate the truth method with the original dataset's training data
#     original_dataset_calibration_results = ttlm.calibrate_truth_method(dataset_name, model, truth_methods, tokenizer = tokenizer, correctness_evaluator = model_judge, 
#     size_of_data = original_dataset_calibration_size, wandb_run = wandb_run, return_method_details = True, seed = seed, max_new_tokens = max_new_tokens, pad_token_id = pad_token_id, do_sample=do_sample) #thresholds are set here maximizing f1 score
# else:
#     original_dataset_calibration_results = ttlm.calibrate_truth_method(dataset_name, model, truth_methods, tokenizer = tokenizer, correctness_evaluator = model_judge, 
#     size_of_data = original_dataset_calibration_size, wandb_run = wandb_run, return_method_details = True, seed = seed)

# #save original_dataset_calibration_results with pickle
# save_name = wandb_name.replace('/','_')
# with open(f'results/original_dataset_calibration_results_{save_name}.pkl', 'wb') as f:
#     pickle.dump(original_dataset_calibration_results, f)

save_name = wandb_name.replace('/','_')

if type(model) != str:
    results = ttlm.evaluate_truth_method(dataset_name, model, truth_methods=truth_methods, 
    eval_metrics = eval_metrics, tokenizer = tokenizer, size_of_data = dataset_size, correctness_evaluator = model_judge, wandb_run = wandb_run, 
    return_method_details = True,  batch_generation = True, wandb_push_method_details = False,
    max_new_tokens = max_new_tokens, pad_token_id = pad_token_id,  previous_context = chat, do_sample=do_sample, user_prompt = prompt_template)
else:
    results = ttlm.evaluate_truth_method(dataset_name, model, truth_methods=truth_methods, 
    eval_metrics = eval_metrics, tokenizer = tokenizer, size_of_data = dataset_size, correctness_evaluator = model_judge, wandb_run = wandb_run, 
    return_method_details = True, wandb_push_method_details = False, previous_context = chat, user_prompt = prompt_template) 


# if type(model) != str:
#     results = ttlm.evaluate_truth_method(dataset_name, model, truth_methods=[semantic_entropy, confidence, entropy, matrix_degree_uncertainty, eccentricity_uncertainty, num_semantic_set_uncertainty, sum_eigen_uncertainty, sentSAR, p_true, inside], 
#     eval_metrics = ['auroc', 'auprc', 'auarc','accuracy'], tokenizer = tokenizer, size_of_data = dataset_size, correctness_evaluator = model_judge, wandb_run = wandb_run, 
#     return_method_details = True, previous_context = chat, batch_generation = True, wandb_push_method_details = False,
#     max_new_tokens = max_new_tokens, pad_token_id = pad_token_id, do_sample = False)
# else:
#     results = ttlm.evaluate_truth_method(dataset_name, model, truth_methods=[semantic_entropy, confidence, entropy, matrix_degree_uncertainty, eccentricity_uncertainty, num_semantic_set_uncertainty, sum_eigen_uncertainty, sentSAR], 
#     eval_metrics = ['auroc', 'auprc', 'auarc','accuracy'], tokenizer = tokenizer, size_of_data = dataset_size, correctness_evaluator = model_judge, wandb_run = wandb_run, 
#     return_method_details = True, wandb_push_method_details = False, previous_context = chat)

method_names = []
score_names =  []
for i in range(len(results['output_dict']['truth_methods'])):
    method_names.append(f'truth_method_{i}_{str(truth_methods[i].__class__.__name__)}')

for key in results['eval_list'][0].keys():
    score_names.append(key)

score_matrix = []
for i in range(len(results['output_dict']['truth_methods'])):
    score_matrix.append([])
    for score_name in score_names:
        score_matrix[i].append(results['eval_list'][i][score_name].round(3))

df = pd.DataFrame(score_matrix, columns = score_names, index = method_names)
output_file = f'evaluations/evaluations_{save_name}.xlsx'

df.to_excel(output_file, sheet_name="Benchmark with Context Results")

#save eval_dict with pickle
save_name = wandb_name.replace('/','_')
with open(f'results/eval_dict_{save_name}.pkl', 'wb') as f:
    pickle.dump(results, f)