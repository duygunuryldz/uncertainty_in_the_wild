

python benchmark_with_adversarial.py --dataset_name trivia_qa --model_name meta-llama/Meta-Llama-3-8B-Instruct --device 0 --dataset_size 1000 --number_of_generations 5 --seed 0 --model_judge gpt-4o-mini --certainty_prompt_index 0
python benchmark_with_adversarial.py --dataset_name trivia_qa --model_name meta-llama/Meta-Llama-3-8B-Instruct --device 0 --dataset_size 1000 --number_of_generations 5 --seed 1 --model_judge gpt-4o-mini --certainty_prompt_index 0
python benchmark_with_adversarial.py --dataset_name trivia_qa --model_name meta-llama/Meta-Llama-3-8B-Instruct --device 0 --dataset_size 1000 --number_of_generations 5 --seed 2 --model_judge gpt-4o-mini --certainty_prompt_index 0
python benchmark_with_adversarial.py --dataset_name trivia_qa --model_name meta-llama/Meta-Llama-3-8B-Instruct --device 0 --dataset_size 1000 --number_of_generations 5 --seed 3 --model_judge gpt-4o-mini --certainty_prompt_index 0
python benchmark_with_adversarial.py --dataset_name trivia_qa --model_name meta-llama/Meta-Llama-3-8B-Instruct --device 0 --dataset_size 1000 --number_of_generations 5 --seed 4 --model_judge gpt-4o-mini --certainty_prompt_index 0


python benchmark_with_adversarial.py --dataset_name gsm8k --model_name meta-llama/Meta-Llama-3-8B-Instruct --device 6 --dataset_size 1000 --number_of_generations 5 --seed 0 --model_judge gpt-4o-mini --certainty_prompt_index 0
python benchmark_with_adversarial.py --dataset_name gsm8k --model_name meta-llama/Meta-Llama-3-8B-Instruct --device 6 --dataset_size 1000 --number_of_generations 5 --seed 1 --model_judge gpt-4o-mini --certainty_prompt_index 0
python benchmark_with_adversarial.py --dataset_name gsm8k --model_name meta-llama/Meta-Llama-3-8B-Instruct --device 6 --dataset_size 1000 --number_of_generations 5 --seed 2 --model_judge gpt-4o-mini --certainty_prompt_index 0
python benchmark_with_adversarial.py --dataset_name gsm8k --model_name meta-llama/Meta-Llama-3-8B-Instruct --device 6 --dataset_size 1000 --number_of_generations 5 --seed 3 --model_judge gpt-4o-mini --certainty_prompt_index 0
python benchmark_with_adversarial.py --dataset_name gsm8k --model_name meta-llama/Meta-Llama-3-8B-Instruct --device 6 --dataset_size 1000 --number_of_generations 5 --seed 4 --model_judge gpt-4o-mini --certainty_prompt_index 0



python benchmark_with_adversarial.py --dataset_name trivia_qa --model_name  gpt-4o-mini --device 6 --dataset_size 1000 --original_dataset_calibration_size 3 --number_of_generations 5 --seed 0 --model_judge gpt-4o-mini --certainty_prompt_index 10
python benchmark_with_adversarial.py --dataset_name trivia_qa --model_name  gpt-4o-mini --device 1 --dataset_size 1000 --original_dataset_calibration_size 3 --number_of_generations 5 --seed 1 --model_judge gpt-4o-mini --certainty_prompt_index 10
python benchmark_with_adversarial.py --dataset_name trivia_qa --model_name  gpt-4o-mini --device 1 --dataset_size 1000 --original_dataset_calibration_size 3 --number_of_generations 5 --seed 2 --model_judge gpt-4o-mini --certainty_prompt_index 10
python benchmark_with_adversarial.py --dataset_name trivia_qa --model_name  gpt-4o-mini --device 1 --dataset_size 1000 --original_dataset_calibration_size 3 --number_of_generations 5 --seed 3 --model_judge gpt-4o-mini --certainty_prompt_index 10
python benchmark_with_adversarial.py --dataset_name trivia_qa --model_name  gpt-4o-mini --device 1 --dataset_size 1000 --original_dataset_calibration_size 3 --number_of_generations 5 --seed 4 --model_judge gpt-4o-mini --certainty_prompt_index 10



python benchmark_with_adversarial.py --dataset_name gsm8k  --model_name  gpt-4o-mini --device 1 --dataset_size 1000 --original_dataset_calibration_size 3 --number_of_generations 5 --seed 0 --model_judge gpt-4o-mini --certainty_prompt_index 10
python benchmark_with_adversarial.py --dataset_name gsm8k  --model_name  gpt-4o-mini --device 1 --dataset_size 1000 --original_dataset_calibration_size 3 --number_of_generations 5 --seed 1 --model_judge gpt-4o-mini --certainty_prompt_index 10
python benchmark_with_adversarial.py --dataset_name gsm8k  --model_name  gpt-4o-mini --device 1 --dataset_size 1000 --original_dataset_calibration_size 3 --number_of_generations 5 --seed 2 --model_judge gpt-4o-mini --certainty_prompt_index 10
python benchmark_with_adversarial.py --dataset_name gsm8k  --model_name  gpt-4o-mini --device 1 --dataset_size 1000 --original_dataset_calibration_size 3 --number_of_generations 5 --seed 3 --model_judge gpt-4o-mini --certainty_prompt_index 10
python benchmark_with_adversarial.py --dataset_name gsm8k  --model_name  gpt-4o-mini --device 1 --dataset_size 1000 --original_dataset_calibration_size 3 --number_of_generations 5 --seed 4 --model_judge gpt-4o-mini --certainty_prompt_index 10


