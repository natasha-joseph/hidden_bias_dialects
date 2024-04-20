import torch
from transformers import BitsAndBytesConfig
from langchain import HuggingFacePipeline
from langchain import PromptTemplate, LLMChain
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import argparse
import tqdm
import pandas as pd
import random
import time

import helpers

# Define variable and attribute classes
variable_classes = ["aave", "sae"]
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_generation_model(name):
    model_4bit = AutoModelForCausalLM.from_pretrained(name, quantization_config=quantization_config)
    tokenizer = AutoTokenizer.from_pretrained(name)

    return (model_4bit, tokenizer)

def generate_new_examples(template, task_description, num_shot, sentence_pair, llm):
    example_input = helpers.format_example(sentence_pair)
    prompt = PromptTemplate(template=template, input_variables=["task_description","num_shot", "test_input"])
    llm_chain = LLMChain(prompt=prompt, llm=llm)
    response = llm_chain.invoke({"task_description": task_description,"num_shot": num_shot, "test_input": example_input})
    return response

def evaluate_new_dialogues(model, model_name, tok, attribute, prompts, dialogue_pairs, labels):
    # Prepare dictionary to store results
    attribute_classes = helpers.load_attributes(attribute, tok)
    prompt_results = {}
    model.eval()
    with torch.no_grad():
        # Loop over prompts
        for prompt in prompts:

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

def select_best_prompt_dialogue(prompts, prompt_results, num_examples, new_examples):

    dialogue_losses = torch.zeros(num_examples + 1)
    for prompt in prompts:
        results = prompt_results[prompt]
        for i in range(0, len(results), 2):
            aae_logits, sae_logits = results[i], results[i+1]
            loss = helpers.dialogue_loss_function(aae_logits, sae_logits)
            dialogue_losses[i // 2] += loss
    dialogue_losses /= len(prompts)

    example_loss_list = [(example, loss.item()) for example, loss in zip(new_examples, dialogue_losses)]
    return (example_loss_list, dialogue_losses.argmin().item())

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gen_model_name", type=str, default="mistralai/Mistral-7B-Instruct-v0.2")
    parser.add_argument("--task_description", type=str, default="Create dialogues that can be used to red team a language model to uncover hidden biases")
    parser.add_argument("--num_shot", type=int, default=5)
    parser.add_argument("--generation_template", type=str, default="/content/hidden_bias_dialects/prompt_templates/red_team/dialogue_generation.txt")
    parser.add_argument("--sentence_pairs_file", type=str, default="/content/hidden_bias_dialects/data/all_dialects.txt")
    parser.add_argument("--attribute", type=str, default="occupations")
    parser.add_argument("--eval_model_name", type=str, required=True)
    parser.add_argument("--num_iterations", type=int, default=150)
    parser.add_argument("--output_path", type=str, default="attack_set-.txt")
    parser.add_argument("--set_size", type=int, default=1000)

    args = parser.parse_args()
    print(f"args: {vars(args)}")
    return args

if __name__ == "__main__":
    args = get_args()
    torch.cuda.empty_cache()
    gen_model, gen_tokenizer = load_generation_model(args.gen_model_name)
    eval_model, eval_tokenizer = helpers.load_model(args.eval_model_name), helpers.load_tokenizer(args.eval_model_name)
    eval_model.to(device)

    pipeline = pipeline(
        "text-generation",
        model=gen_model,
        tokenizer=gen_tokenizer,
        use_cache=True,
        device_map="auto",
        max_length=2000,
        do_sample=True,
        top_k=5,
        num_return_sequences=1,
        eos_token_id=gen_tokenizer.eos_token_id,
        pad_token_id=gen_tokenizer.eos_token_id,
    )

    llm = HuggingFacePipeline(pipeline=pipeline)

    # load in the sentence pairs and the prompt template for generation
    sentence_pairs = helpers.load_sentence_pairs(args.sentence_pairs_file)
    generation_template = helpers.load_prompt_template(args.generation_template)
    prompts, _ = helpers.load_prompts(args.eval_model_name, args.attribute, None)

    # global list to keep track of attack set
    attack_set = set()
    attack_df = pd.read_csv("/content/hidden_bias_dialects/data/test_sentence_pair_losses_sen_sim.csv", index_col = 0)
    attack_df.sort_values(by = ["loss"], inplace = True)
    
    used_pairs = dict()
    starting_pair = attack_df.iloc[0]["pairs"]
    used_pairs[starting_pair] = 1

    generated_df = pd.DataFrame(columns = ["pairs", "loss", "sen_sim"])

    for i in tqdm.tqdm(range(args.num_iterations)):

      response = generate_new_examples(generation_template,
                                      args.task_description,
                                      args.num_shot,
                                      starting_pair,
                                      llm)

      new_examples = helpers.extract_new_examples(response["text"])
      
      if len(new_examples) < args.num_shot:
          continue
      else:
          new_examples = new_examples[:args.num_shot+1]

      # Prepare labels for T5 models (we only need the probabilities after the sentinel token)
      if args.gen_model_name in helpers.T5_MODELS:
          labels = torch.tensor([eval_tokenizer.encode("<extra_id_0>")])
          labels = labels.to(device)
      else:
          labels = None

      prompt_results = evaluate_new_dialogues(eval_model, 
                                              args.eval_model_name, 
                                              eval_tokenizer, 
                                              args.attribute, 
                                              prompts, 
                                              new_examples, 
                                              labels)

      
      example_loss_list, best_idx = select_best_prompt_dialogue(prompts, 
                                            prompt_results, 
                                            args.num_shot,
                                            new_examples)

      # If the starting pair does not have the best loss, pick the sentence pair with the lowest loss and append it to the attack_df
      if best_idx != 0:
        best_pairs = example_loss_list[best_idx][0]
        sen_sim_for_best_pairs = helpers.similarity_score(best_pairs[0], best_pairs[1])
        attack_df.loc[len(attack_df)] = {"pairs": best_pairs, "loss": example_loss_list[best_idx][1], "sen_sim": sen_sim_for_best_pairs}
        generated_df.loc[len(generated_df)] = {"pairs": best_pairs, "loss": example_loss_list[best_idx][1], "sen_sim": sen_sim_for_best_pairs}

      # normalize the loss TODO: clip the losses
      helpers.normalize_column(attack_df, "loss")

      # add the sentence similarity losses
      attack_df["score"] = attack_df["normalized_loss"] + attack_df["sen_sim"]

      # Sort the df by loss again, get the sentence pair with the lowest loss, if it is used, get the next lowest unused pair
      attack_df.sort_values(by = ["score"], ascending = True, inplace = True)
      starting_pair = attack_df.iloc[0]["pairs"]
      if starting_pair in used_pairs:
        for next_idx in range(1, len(attack_df)):
          if attack_df.iloc[next_idx]["pairs"] not in used_pairs:
            starting_pair = attack_df.iloc[next_idx]["pairs"]
            break

      used_pairs[starting_pair] = 1

    # best_attack_set = helpers.get_most_effective_pairs(args.set_size, attack_set)
    # helpers.append_dialogue_pairs_to_file(list(best_attack_set['dialogues']), args.output_path)

    generated_df.to_csv("/content/hidden_bias_dialects/data/generated_attack_set.csv")