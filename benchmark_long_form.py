import os 
import sys

sys.path.append('../TruthTorchLM/src/')
import TruthTorchLM as ttlm
import TruthTorchLM.long_form_generation as LFG

import json
import wandb
import torch
import pickle
import random
import argparse
import pandas as pd

from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import DebertaForSequenceClassification, DebertaTokenizer
from sentence_transformers.cross_encoder import CrossEncoder



os.environ['SERPER_API_KEY'] = 'YOUR_KEY'
os.environ["OPENAI_API_KEY"] = 'YOUR_KEY'
os.environ["CUDA_VISIBLE_DEVICES"] = "5, 6"

args = argparse.ArgumentParser()
args.add_argument("--model_name", type=str, default='meta-llama/Llama-2-7b-chat-hf')
args.add_argument("--device", type=int, default = 0)
args.add_argument("--dataset_name", type=str, default="longfact_concepts")
args.add_argument("--dataset_size", type=int, default=100)
args.add_argument('--calibration_data_name', type=str, default="trivia_qa")
args.add_argument('--calibration_data_size', type=int, default=100)

args.add_argument('--claim_chk', type=str, default="qag", choices=['qg', 'qag', 'ace', 'naive'])
args.add_argument('--agg_strategy', type=str, default="max", choices=['min', 'max', 'avg']) #truth value aggregation strategy for claim check methods where num_questions > 1

args.add_argument("--number_of_generations", type=int, default=5)
args.add_argument("--number_of_questions", type=int, default=1)
args.add_argument("--number_of_answers", type=int, default=1)
args.add_argument('--model_judge', type=str, default='gpt-4o-mini')
args.add_argument('--qgen_model', type=str, default='gpt-4o-mini') #used in decomposition and claim check
args.add_argument("--seed", type=int, default=0)

args.add_argument('--save_name', type=str, default='lgf')


args = args.parse_args()

qgen_model = args.qgen_model
model_name = args.model_name
device = args.device
dataset_name = args.dataset_name
dataset_size = args.dataset_size
calibration_data_name = args.calibration_data_name
calibration_data_size = args.calibration_data_size

number_of_generations = args.number_of_generations
number_of_questions = args.number_of_questions
number_of_answers = args.number_of_answers
seed = args.seed

claim_chk = args.claim_chk

max_new_tokens = 64

#Adjust Seed
random.seed(seed)
torch.manual_seed(seed)

#Create wandb run
data_save_name = "longfact_objects" if "longfact_objects" in dataset_name else dataset_name
data_save_name = "longfact_concepts" if "longfact_concepts" in data_save_name else data_save_name
data_save_name = "factscore" if "factscore" in data_save_name else data_save_name
model_save_name = "Llama-3-8B" if "Llama-3-8B" in model_name else model_name
qmodel_save_name = "Llama-3-8B" if "Llama-3-8B" in qgen_model else qgen_model
qmodel_save_name = "" if args.claim_chk == "naive" else qmodel_save_name
wandb_name = f'{args.save_name}_{seed}_{data_save_name}_{model_save_name}_{qmodel_save_name}_{claim_chk}_{seed}_{dataset_size}'
if args.claim_chk == "ace":
    wandb_name += f"__Q{number_of_questions}_A{number_of_answers}"
elif args.claim_chk == "qag" or args.claim_chk == "qg":
    if number_of_questions > 1:
        wandb_name += f"__{args.agg_strategy}_{number_of_questions}"
    else:
        wandb_name += f"__{number_of_questions}"
wandb_run = wandb.init(
      project="uncertainty_in_the_wild_LFG",
      name= wandb_name,
      config=args
      )

api_models = ["gpt-3.5-turbo", "gpt-4o-mini"]
device = torch.device("cuda:" + str(device) if torch.cuda.is_available() else "cpu")
#Create main model
if model_name in api_models:
    model = model_name
    tokenizer = None
    pad_token_id = None
else:
    # model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16).to(device)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    pad_token_id = tokenizer.pad_token_id
    if pad_token_id == None:
        pad_token_id = model.config.eos_token_id

#Create Question Generation model
if qgen_model in api_models:
    qgen_model = qgen_model
    qgen_tokenizer = None
elif qgen_model == model_name:
    qgen_model = model
    qgen_tokenizer = tokenizer
else:
    qgen_model = AutoModelForCausalLM.from_pretrained(qgen_model, torch_dtype=torch.float16).to(device)
    qgen_tokenizer = AutoTokenizer.from_pretrained(qgen_model)

#Many Truth methods utilize entailment model
model_for_entailment = DebertaForSequenceClassification.from_pretrained('microsoft/deberta-large-mnli').to(device)
tokenizer_for_entailment = DebertaTokenizer.from_pretrained('microsoft/deberta-large-mnli')

model_for_similarity = CrossEncoder('cross-encoder/stsb-roberta-large', num_labels=1, device="cuda")

#Define Decomposition Method
decomposition_method = LFG.decomposition_methods.StructuredDecompositionAPI('gpt-4o-mini', decomposition_depth=1)

#Create Truth Method Objects
semantic_entropy = ttlm.truth_methods.SemanticEntropy(ttlm.scoring_methods.LengthNormalizedScoring(), number_of_generations=number_of_generations, model_for_entailment=model_for_entailment, tokenizer_for_entailment=tokenizer_for_entailment)
confidence = ttlm.truth_methods.Confidence(ttlm.scoring_methods.LengthNormalizedScoring())
entropy = ttlm.truth_methods.Entropy(ttlm.scoring_methods.LengthNormalizedScoring(), number_of_generations=number_of_generations)
p_true = ttlm.truth_methods.PTrue(number_of_ideas=number_of_generations)
matrix_degree_uncertainty = ttlm.truth_methods.MatrixDegreeUncertainty(number_of_generations=number_of_generations, model_for_entailment=model_for_entailment, tokenizer_for_entailment=tokenizer_for_entailment)
eccentricity_uncertainty = ttlm.truth_methods.EccentricityUncertainty(number_of_generations=number_of_generations, model_for_entailment=model_for_entailment, tokenizer_for_entailment=tokenizer_for_entailment)
sum_eigen_uncertainty = ttlm.truth_methods.SumEigenUncertainty(number_of_generations=number_of_generations, model_for_entailment=model_for_entailment, tokenizer_for_entailment=tokenizer_for_entailment)
sentSAR = ttlm.truth_methods.SentSAR(number_of_generations=number_of_generations, model_for_similarity=model_for_similarity)
sar = ttlm.truth_methods.SAR(number_of_generations=number_of_generations, model_for_similarity=model_for_similarity)
inside = ttlm.truth_methods.Inside(number_of_generations=number_of_generations)
self_detection = ttlm.truth_methods.SelfDetection(number_of_questions = number_of_generations, method_for_similarity='semantic', model_for_entailment=model_for_entailment, tokenizer_for_entailment=tokenizer_for_entailment)
kernel_entropy = ttlm.truth_methods.KernelLanguageEntropy(number_of_generations=number_of_generations, model_for_entailment=model_for_entailment, tokenizer_for_entailment=tokenizer_for_entailment)
lars = ttlm.truth_methods.LARS(device = device)
eccentricity_confidence = ttlm.truth_methods.EccentricityConfidence(number_of_generations=number_of_generations, model_for_entailment=model_for_entailment, tokenizer_for_entailment=tokenizer_for_entailment)
matrix_degree_confidence = ttlm.truth_methods.MatrixDegreeConfidence(number_of_generations=number_of_generations, model_for_entailment=model_for_entailment, tokenizer_for_entailment=tokenizer_for_entailment)
attention_score = ttlm.truth_methods.AttentionScore(layer_index=23)
mars = ttlm.truth_methods.MARS(device=device, mars_temperature = 0.1)
verbalized_confidence = ttlm.truth_methods.VerbalizedConfidence()

if type(model) != str:
    with open('meta-llama_Meta-Llama-3-8B-Instruct_trivia_qa_gsm8k_[8000, 5000]_saplma_output_dict.pkl', 'rb') as f:
        saplma_dict = pickle.load(f)

    saplma_model = saplma_dict['model']
    saplma = ttlm.truth_methods.SAPLMA(saplma_model=saplma_model, layer_index = -1)

if type(model) != str:
    truth_methods = [semantic_entropy, confidence, entropy, matrix_degree_uncertainty, eccentricity_uncertainty, sum_eigen_uncertainty, sentSAR, p_true, self_detection, kernel_entropy, lars, inside, mars, attention_score, matrix_degree_confidence, eccentricity_confidence, saplma, verbalized_confidence, sar]
else:
    truth_methods = [semantic_entropy, confidence, entropy, matrix_degree_uncertainty, eccentricity_uncertainty, sum_eigen_uncertainty, sentSAR, self_detection, kernel_entropy, lars, mars, matrix_degree_confidence, eccentricity_confidence, verbalized_confidence, sar]


#Create Claim Check Method Object
if claim_chk == 'qag':
    claim_check_mthd = LFG.claim_check_methods.QuestionAnswerGeneration( model=qgen_model, tokenizer=qgen_tokenizer, num_questions=number_of_questions, max_answer_trials=2,
                                                                    truth_methods=truth_methods, seed=args.seed, entailment_model=model_for_entailment, entailment_tokenizer=tokenizer_for_entailment)
elif claim_chk == 'qg':
    if type(model) == str:
        truth_methods = [semantic_entropy, entropy, matrix_degree_uncertainty, eccentricity_uncertainty, sum_eigen_uncertainty, sentSAR, self_detection, kernel_entropy, matrix_degree_confidence, eccentricity_confidence, verbalized_confidence, sar]
    
    claim_check_mthd = LFG.claim_check_methods.QuestionGeneration( model=qgen_model, tokenizer=qgen_tokenizer, num_questions=number_of_questions, 
                                                                      truth_methods=truth_methods, seed=args.seed)
elif claim_chk == 'ace':
    claim_check_mthd = LFG.claim_check_methods.AnswerClaimEntailment( model=qgen_model, tokenizer=qgen_tokenizer, 
                                                                    num_questions=number_of_questions, num_answers_per_question=number_of_answers,
                                                                    entailment_model=model_for_entailment, entailment_tokenizer=tokenizer_for_entailment,
                                                                    seed=args.seed)
elif claim_chk == 'naive':
    if type(model) == str:
        truth_methods = [semantic_entropy, entropy, matrix_degree_uncertainty, eccentricity_uncertainty, sum_eigen_uncertainty, sentSAR, self_detection, kernel_entropy, matrix_degree_confidence, eccentricity_confidence, verbalized_confidence, sar]

    claim_check_mthd = LFG.claim_check_methods.NaiveApplication(truth_methods=truth_methods)

#Create claim evaluator object
safe = LFG.ClaimEvaluator(rater='gpt-4o-mini', tokenizer = None, max_steps = 2, max_retries = 2, num_searches = 2)

#Define Metrics
sample_level_eval_metrics = ['accuracy', 'f1', 'precision', 'recall']
dataset_level_eval_metrics = ['auroc', 'auprc', 'auarc','accuracy', 'f1', 'precision', 'recall', 'prr']

#Create dataset
if dataset_name == "factscore":
    questions = []
    with open('/home/yavuz/yavuz/data/labeled/ChatGPT.jsonl', 'r') as file:
        for line in file:
            questions.append(json.loads(line))
    label_mapping = {"NS": 0, "S": 1, "IR":-1}
    data = []
    for question in questions:
        if question['annotations']:
            claims = []
            labels = []
            for annotation in question['annotations']:
                if annotation['human-atomic-facts']:
                    for facts in annotation['human-atomic-facts']:
                        if label_mapping[facts['label']] != -1:
                            claims.append(facts['text'])
                            labels.append(label_mapping[facts['label']])
            data.append({'question':question['input'], "generated_tex":question['output'],
                        "claims":claims, "claim_correctness":labels})
    if dataset_size < len(data):
        data = random.sample(data, dataset_size)
    user_prompt = "{question_context}"
elif ".pkl" in dataset_name:
    file = open("./LFG_data/" + dataset_name,'rb')
    data = pickle.load(file)
    if dataset_size < len(data):
        data = random.sample(data, dataset_size)
    user_prompt = "Question: {question_context}"
else:
    data = dataset_name
    user_prompt = "Question: {question_context}"


#Run evaluation
if type(model) != str:
    results = LFG.evaluate_truth_method_long_form(dataset=data, model=model, tokenizer=tokenizer,
                                    sample_level_eval_metrics=sample_level_eval_metrics, dataset_level_eval_metrics=dataset_level_eval_metrics,
                                    decomp_method=decomposition_method, claim_check_methods=[claim_check_mthd],
                                    claim_evaluator = safe, size_of_data=dataset_size,  previous_context=[{'role': 'system', 'content': 'You are a helpful assistant. Give precise answers.'}], 
                                    user_prompt=user_prompt, seed=args.seed,  return_method_details = True, return_calim_eval_details=True, wandb_run = wandb_run,  
                                    add_generation_prompt = True, continue_final_message = False,
                                    pad_token_id = pad_token_id)
else:
    results = LFG.evaluate_truth_method_long_form(dataset=data, model=model, tokenizer=tokenizer,
                                    sample_level_eval_metrics=sample_level_eval_metrics, dataset_level_eval_metrics=dataset_level_eval_metrics,
                                    decomp_method=decomposition_method, claim_check_methods=[claim_check_mthd],
                                    claim_evaluator = safe, size_of_data=dataset_size,  previous_context=[{'role': 'system', 'content': 'You are a helpful assistant. Give precise answers.'}], 
                                    user_prompt=user_prompt, seed=args.seed,  return_method_details = True, return_calim_eval_details=True, wandb_run = wandb_run,  
                                    add_generation_prompt = True, continue_final_message = False)


#Save Results
save_name = wandb_name.replace('/','_')
df = pd.DataFrame(results['dataset_level_eval_list']).T
output_file = f'evaluations/evaluations_dataset_level_{save_name}.xlsx'
df.to_excel(output_file, sheet_name="LFG Benchmark Results")

df = pd.DataFrame(results['sample_level_eval_list']).T
output_file = f'evaluations/evaluations_sample_level_{save_name}.xlsx'
df.to_excel(output_file, sheet_name="LFG Benchmark Results")

#save eval_dict with pickle
with open(f'results/eval_dict_LFG_{save_name}.pkl', 'wb') as f:
    pickle.dump(results, f)