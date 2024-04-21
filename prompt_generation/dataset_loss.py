import torch
from torch.cuda import list_gpu_processes
from transformers import BitsAndBytesConfig
from langchain import HuggingFacePipeline
from langchain import PromptTemplate, LLMChain
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import argparse
import tqdm
import pandas as pd

import helpers
from transformers.models.speech_to_text.tokenization_speech_to_text import sentencepiece

# Define variable and attribute classes
variable_classes = ["aave", "sae"]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def evaluate_new_dialogues(model, model_name, tok, attribute, prompts, dialogue_pairs, labels):
    # Prepare dictionary to store results
    attribute_classes = helpers.load_attributes(attribute, tok)
    prompt_results = {}
    model.eval()
    with torch.no_grad():
        # Loop over prompts
        for prompt in tqdm.tqdm(prompts):

            # Compute prompt-specific results
            results = []
            for variable_pair in dialogue_pairs:
                variable_0, variable_1 = variable_pair[0], variable_pair[1]

                # Pass prompts through model and select attribute probabilities
                for i, variable in enumerate([variable_0, variable_1]):
                    probs_attribute = helpers.get_attribute_probs(
                        prompt.format(variable),
                        attribute_classes,
                        model,
                        model_name,
                        tok,
                        device,
                        labels
                    )

                    results.append((
                        variable,
                        variable_classes[i],
                        attribute_classes,
                        probs_attribute
                    ))

            # Add results to dictionary
            prompt_results[prompt] = results

    return prompt_results

def get_losses(prompts, prompt_results, sentence_pairs, num_positive_attributes = 41):
  
    dialogue_losses = torch.zeros(len(sentence_pairs))
    for prompt in tqdm.tqdm(prompts):
        results = prompt_results[prompt]
        for i in range(0, len(results), 2):
            aae_logits, sae_logits = results[i], results[i+1]
            loss = helpers.dialogue_loss_function(aae_logits, sae_logits, num_positive_attributes)
            dialogue_losses[i // 2] += loss
    dialogue_losses /= len(prompts)

    example_loss_list = [(example, loss.item()) for example, loss in zip(sentence_pairs, dialogue_losses)]
    return example_loss_list

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gen_model_name", type=str, default="mistralai/Mistral-7B-Instruct-v0.2")
    parser.add_argument("--sentence_pairs_file", type=str, default="/content/hidden_bias_dialects/data/all_dialects.txt")
    parser.add_argument("--attribute", type=str, default="occupations")
    parser.add_argument("--eval_model_name", type=str, required=True)
    parser.add_argument("--output_path", type=str, default="attack_set-.txt")
    parser.add_argument("--set_size", type=int, default=1000)

    args = parser.parse_args()
    print(f"args: {vars(args)}")
    return args

if __name__ == "__main__":
    args = get_args()
    torch.cuda.empty_cache()
    eval_model, eval_tokenizer = helpers.load_model(args.eval_model_name), helpers.load_tokenizer(args.eval_model_name)
    eval_model.to(device)

    # load in the sentence pairs and the prompt template for generation
    sentence_pairs = helpers.load_sentence_pairs(args.sentence_pairs_file)
    sentence_pairs = [sentence_pair[:2] for sentence_pair in sentence_pairs]

    prompts, _ = helpers.load_prompts(args.eval_model_name, args.attribute, None)

    # Prepare labels for T5 models (we only need the probabilities after the sentinel token)
    if args.gen_model_name in helpers.T5_MODELS:
        labels = torch.tensor([eval_tokenizer.encode("<extra_id_0>")])
        labels = labels.to(device)
    else:
        labels = None

    # Get the loss for each dialogue
    prompt_results = evaluate_new_dialogues(eval_model, 
                                            args.eval_model_name, 
                                            eval_tokenizer, 
                                            args.attribute, 
                                            prompts, 
                                            sentence_pairs, 
                                            labels)


    if args.attribute == "occupations":
      losses = get_losses(prompts, prompt_results, sentence_pairs, 41)
    elif args.attribute == "valence":
      losses = get_losses(prompts, prompt_results, sentence_pairs, 280)
    
    loss_df = pd.DataFrame(losses, columns = ['pairs', 'loss'])
    loss_df.sort_values(by = ['loss'])
    loss_df.to_csv("/content/hidden_bias_dialects/data/valence_sentence_pair_losses.csv")


