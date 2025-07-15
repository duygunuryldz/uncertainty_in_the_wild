import os 
import sys
sys.path.append('../TruthTorchLM/src/')
#os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6"
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
from TruthTorchLM.templates import DEFAULT_SYSTEM_PROMPT, DEFAULT_USER_PROMPT, SELF_DETECTION_QUESTION_PROMPT, SELF_DETECTION_SYSTEM_PROMPT, ENTAILMENT_PROMPT
from TruthTorchLM.templates import PTRUE_SYSTEM_PROMPT, PTRUE_USER_PROMPT, PTRUE_MODEL_OUTPUT
import pandas as pd
import random
import string





args = argparse.ArgumentParser()
args.add_argument("--model_name", type=str, default='meta-llama/Llama-2-7b-chat-hf')
args.add_argument("--device", type=int, default = 0)
args.add_argument("--dataset_name", type=str, default='trivia_qa')
args.add_argument("--dataset_size", type=int, default=1000)
args.add_argument('--original_dataset_calibration_size', type=int, default=100)

args.add_argument("--number_of_generations", type=int, default=5)
args.add_argument("--seed", type=int, default=0)
args.add_argument('--model_judge', type=str, default='gpt-4o-mini')

args.add_argument('--calibration_dataset', type=str, default='trivia_qa')
args.add_argument('--calibration_dataset_size', type=int, default=100)

args.add_argument('--save_name', type=str, default='benchmark_typo')
args.add_argument('--auto_device', action='store_true', help="Enable auto device")
args.add_argument('--num_typos', type=int, default=1)

args = args.parse_args()


number_of_generations = args.number_of_generations
seed = args.seed
model_name = args.model_name
device = args.device
dataset_name = args.dataset_name
dataset_size = args.dataset_size
max_new_tokens = 128
original_dataset_calibration_size = args.original_dataset_calibration_size
num_typos = args.num_typos
calibration_dataset = args.calibration_dataset
calibration_dataset_size = args.calibration_dataset_size

device = torch.device("cuda:" + str(device) if torch.cuda.is_available() else "cpu")
wandb_name = f'{args.save_name}_{dataset_name}_{model_name}_{dataset_size}_{number_of_generations}_{seed}_{calibration_dataset}_{calibration_dataset_size}_{num_typos}'

wandb_run = wandb.init(
      # Set the project where this run will be logged
      project="uncertainty_in_the_wild_typo_gpt_fixed",
      # We pass a run name (otherwise it’ll be randomly assigned, like sunshine-lollypop-10)
      name= wandb_name,
      config={"seed": seed}
      )



def simulate_typo(s, 
                  num_typos=1, 
                  weights=None):
    """
    Introduce typographical errors into a string.
    
    Parameters:
        s (str): The original string.
        num_typos (int): How many typo operations to apply.
        weights (dict): A dictionary specifying the probabilities for each operation. 
                        For example: {'replace':0.4, 'delete':0.2, 'insert':0.3, 'swap':0.1}
                        If None, equal probabilities are assumed.
                        
    Returns:
        str: The modified string with typos introduced.
    """
    
    # Define possible operations
    operations = ["replace", "delete", "insert", "swap"]
    
    # If no weights provided, use equal probability
    if weights is None:
        weights = {op: 1.0/len(operations) for op in operations}
    else:
        # Normalize weights if they do not sum to 1
        total = sum(weights.values())
        weights = {op: w/total for op, w in weights.items()}
    
    # Function to apply a single typo
    def apply_single_typo(text):
        # If text is empty or a single character, a delete or swap might not be possible.
        # We'll handle edge cases gracefully.
        if len(text) <= 1:
            # If it's empty or single char, best we can do is insert.
            operation = "insert"
        else:
            # Choose operation based on given weights
            r = random.random()
            cumulative = 0
            for op in operations:
                cumulative += weights[op]
                if r <= cumulative:
                    operation = op
                    break

        # Pick a random position in the string
        pos = random.randint(0, max(len(text)-1, 0))
        
        if operation == "replace" and len(text) > 0:
            # Replace one character at position pos with a random letter different from the original
            original_char = text[pos]
            candidates = [ch for ch in string.ascii_lowercase if ch != original_char.lower()]
            new_char = random.choice(candidates)
            # Keep the same case
            if original_char.isupper():
                new_char = new_char.upper()
            return text[:pos] + new_char + text[pos+1:]
        
        elif operation == "delete" and len(text) > 0:
            # Delete the character at position pos
            return text[:pos] + text[pos+1:]
        
        elif operation == "insert":
            # Insert a random character at position pos
            new_char = random.choice(string.ascii_lowercase)
            # Small chance to uppercase
            if random.random() < 0.2:
                new_char = new_char.upper()
            return text[:pos] + new_char + text[pos:]
        
        elif operation == "swap" and len(text) > 1:
            # Swap two adjacent characters
            # If pos is the last index, swap with previous character
            if pos == len(text)-1:
                pos -= 1
            char_list = list(text)
            char_list[pos], char_list[pos+1] = char_list[pos+1], char_list[pos]
            return "".join(char_list)
        
        # If none of the above conditions apply (edge cases), just return the original text
        return text

    # Apply the specified number of typos
    new_str = s
    for _ in range(num_typos):
        new_str = apply_single_typo(new_str)
    
    return new_str

# # Example usage:
# original = "Example"
# typo_version = simulate_typo(original, num_typos=3, weights={'replace':0.5, 'delete':0.2, 'insert':0.2, 'swap':0.1})
# print("Original:", original)
# print("With typos:", typo_version)



if 'gpt' in model_name or 'claude' in model_name:
    model = model_name
    tokenizer = None
    pad_token_id = None
else:
    if args.auto_device:
        model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map = 'auto')
    else:
        model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    pad_token_id = tokenizer.pad_token_id
    if pad_token_id == None:
        pad_token_id = model.config.eos_token_id
    if tokenizer.chat_template == None:#set a default chat template
        print('No default chat template found. Setting a default chat template.')
        tokenizer.chat_template = tokenizer.default_chat_template
        print(tokenizer.default_chat_template)

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
cross_examination = ttlm.truth_methods.CrossExamination(model_examiner = model, tokenizer_examiner = tokenizer)
google_search = ttlm.truth_methods.GoogleSearchCheck()
eccentricity_confidence = ttlm.truth_methods.EccentricityConfidence(number_of_generations=number_of_generations, model_for_entailment=model_for_entailment, tokenizer_for_entailment=tokenizer_for_entailment)
matrix_degree_confidence = ttlm.truth_methods.MatrixDegreeConfidence(number_of_generations=number_of_generations, model_for_entailment=model_for_entailment, tokenizer_for_entailment=tokenizer_for_entailment)
attention_score = ttlm.truth_methods.AttentionScore(layer_index=23)
mars = ttlm.truth_methods.MARS(device=device, mars_temperature = 0.1)
verbalized_confidence = ttlm.truth_methods.VerbalizedConfidence()
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



#truth_methods = [eccentricity_confidence, matrix_degree_confidence, attention_score, eccentricity_uncertainty, matrix_degree_uncertainty, saplma]
if 'greedy' in args.save_name:
    do_sample = False
else:
    do_sample = True


eval_metrics = ['auroc', 'auprc', 'auarc','accuracy', 'f1', 'precision', 'recall', 'prr']
model_judge = ttlm.evaluators.ModelJudge(args.model_judge)



#first calibrate the truth method with the original dataset's training data
# if type(model) != str:
#     original_dataset_calibration_results = ttlm.calibrate_truth_method(dataset_name, model, truth_methods, tokenizer = tokenizer, correctness_evaluator = model_judge, 
#     size_of_data = original_dataset_calibration_size, wandb_run = wandb_run, return_method_details = True, seed = seed, max_new_tokens = max_new_tokens, pad_token_id = pad_token_id, wandb_push_method_details = False) #thresholds are set here maximizing f1 score
# else:
#     original_dataset_calibration_results = ttlm.calibrate_truth_method(dataset_name, model, truth_methods, tokenizer = tokenizer, correctness_evaluator = model_judge, 
#     size_of_data = original_dataset_calibration_size, wandb_run = wandb_run, return_method_details = True, seed = seed)

# #save original_dataset_calibration_results with pickle
# save_name = wandb_name.replace('/','_')
# with open(f'results/original_dataset_calibration_results_{save_name}.pkl', 'wb') as f:
#     pickle.dump(original_dataset_calibration_results, f)

#get dataset and apply typo
dataset = ttlm.utils.dataset_utils.get_dataset(dataset_name, size_of_data = dataset_size, seed = seed, split = 'test')

for i in range(len(dataset)):
    dataset[i]['question'] = simulate_typo(dataset[i]['question'], num_typos=args.num_typos)

#print(dataset[:10])


prompt_template = 'You are a helpful assistant. Answer the following question in a single brief but complete sentence. Question: {question_context} Answer:'


if type(model) != str:
    results = ttlm.evaluate_truth_method(dataset, model, truth_methods=truth_methods, 
    eval_metrics = eval_metrics, tokenizer = tokenizer, correctness_evaluator = model_judge, wandb_run = wandb_run, 
    return_method_details = True,  batch_generation = True, wandb_push_method_details = False,
    max_new_tokens = max_new_tokens, pad_token_id = pad_token_id, do_sample = do_sample, seed = seed, user_prompt = prompt_template, previous_context = [])
else:
    results = ttlm.evaluate_truth_method(dataset, model, truth_methods=truth_methods, 
    eval_metrics = eval_metrics, tokenizer = tokenizer, correctness_evaluator = model_judge, wandb_run = wandb_run, 
    return_method_details = True, wandb_push_method_details = False, seed = seed, user_prompt = prompt_template, previous_context = [])  


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

save_name = wandb_name.replace('/','_')
#save eval_dict with pickle
with open(f'results/eval_dict_{save_name}.pkl', 'wb') as f:
    pickle.dump(results, f)

#truth_methods.remove(cross_examination)#this doesn't require calibration

# if calibration_dataset != dataset_name:
#     if type(model) != str:
#         calibration_results = ttlm.calibrate_truth_method(calibration_dataset, model, truth_methods, tokenizer = tokenizer, correctness_evaluator = model_judge, 
#         size_of_data = calibration_dataset_size, wandb_run = wandb_run, return_method_details = True, seed = seed, max_new_tokens = max_new_tokens, pad_token_id = pad_token_id, do_sample = do_sample) #thresholds are set here maximizing f1 score
#     else:
#         calibration_results = ttlm.calibrate_truth_method(calibration_dataset, model, truth_methods, tokenizer = tokenizer, correctness_evaluator = model_judge, 
#         size_of_data = calibration_dataset_size, wandb_run = wandb_run, return_method_details = True, seed = seed)


#     #save calibration_results with pickle
#     with open(f'results/calibration_results_{save_name}.pkl', 'wb') as f:
#         pickle.dump(calibration_results, f)

#     #change normalized truth values of the truth methods with the calibration results
#     for i in range(len(truth_methods)):
#         truth_values = results['output_dict'][f'truth_method_{i}']['truth_values']
#         for j in range(len(truth_values)):
#             results['output_dict'][f'truth_method_{i}']['normalized_truth_values'][j] = truth_methods[i].normalizer(truth_values[j])#normalization with the new threshold and std


#     eval_list = ttlm.evaluators.get_metric_scores(output_dict=results['output_dict'], eval_metrics=eval_metrics, seed=seed)#evaluate the previous results again

#     eval_dict = eval_list[0]
    

#     #push it to wandb
#     for key, _ in eval_dict.items():
#         methods = []
#         scores = []
#         for i, cur_eval_dict in enumerate(eval_list):
#             score = cur_eval_dict[key]
#             scores.append(score)
#             methods.append(str(truth_methods[i].__class__.__name__) + '_after_recalibration')
#             wandb_run.log({f'{key}_of_method_{i}_after_recalibration': score})

#         data = [[method, score] for (method, score) in zip(methods, scores)]
#         table = wandb.Table(data=data, columns = ["methods_after_recalibration", "scores_after_recalibration"])
#         wandb.log({f"{key}_after_calibration" : wandb.plot.bar(table, "methods_after_recalibration", "scores_after_recalibration",
#                         title=f"{key} Scores of Truth Methods after recalibration")}) 


    

#     score_matrix_after_calibration = []
#     for i in range(len(results['output_dict']['truth_methods'])):
#         score_matrix_after_calibration.append([])
#         for score_name in score_names:
#             score_matrix_after_calibration[i].append(eval_list[i][score_name].round(3))
#     # Create DataFrames for each dataset

#     df1 = pd.DataFrame(score_matrix, columns=score_names, index=method_names)
#     df2 = pd.DataFrame(score_matrix_after_calibration, columns=score_names, index=method_names)

#     # Concatenate with MultiIndex columns to differentiate datasets
#     combined_df = pd.concat(
#         {f"Scores in {dataset_name} calibrated on {dataset_name}": df1, f"Scores in {dataset_name} calibrated on {calibration_dataset}": df2},
#         axis=1
#     ) 

#     output_file = f'evaluations/evaluations_{save_name}.xlsx'
#     combined_df.to_excel(output_file, sheet_name="Scores calibration on different datasets")







