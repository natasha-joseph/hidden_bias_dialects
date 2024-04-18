import random
import os
import numpy as np
import torch
from torch.nn import functional as F
import re
from transformers import (
    GPT2LMHeadModel, 
    GPT2Tokenizer, 
    RobertaForMaskedLM, 
    RobertaTokenizer, 
    T5ForConditionalGeneration,
    T5Tokenizer
)

import prompting
random.seed(383)

# Define path to attribute lists
ATTRIBUTES_PATH = os.path.abspath("../data/attributes/{}.txt")

# Define model groups
GPT2_MODELS = ["gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl"]
ROBERTA_MODELS = ["roberta-base", "roberta-large"]
T5_MODELS = ["t5-small", "t5-base", "t5-large", "t5-3b"]

# Function to load pretrained language model
def load_model(model_name):
    if model_name in GPT2_MODELS:
        return GPT2LMHeadModel.from_pretrained(
            model_name 
        )
    elif model_name in ROBERTA_MODELS:
        return RobertaForMaskedLM.from_pretrained(
            model_name
        )
    elif model_name in T5_MODELS:
        return T5ForConditionalGeneration.from_pretrained(
            model_name 
        )
    else:
        raise ValueError(f"Model {model_name} not supported.")
    
# Function to load tokenizer
def load_tokenizer(model_name):
    if model_name in GPT2_MODELS:
        return GPT2Tokenizer.from_pretrained(
            model_name 
        )
    elif model_name in ROBERTA_MODELS:
        return RobertaTokenizer.from_pretrained(
            model_name 
        )
    elif model_name in T5_MODELS:
        return T5Tokenizer.from_pretrained(
            model_name 
        )
    else:
        raise ValueError(f"Model {model_name} not supported.")

# Function to prepare and load prompts
def load_prompts(model_name, attribute, variable):

    # Overt prejudice prompts
    if variable == "race":
        prompts = prompting.RACE_PROMPTS

    # Covert prejudice prompts
    else:
        if attribute == "occupations":
            prompts = prompting.OCCUPATION_PROMPTS
        else:
            raise ValueError(f"Attribute {attribute} not supported.")
      
    # Model-specific preparations
    if model_name in ROBERTA_MODELS:
        prompts = [p + " <mask>" for p in prompts]
    elif model_name in T5_MODELS:
        prompts = [p + " <extra_id_0>" for p in prompts]
    cal_prompts = [p.format("") for p in prompts]
    if model_name == "gpt3":
        prompts = [p + " {{}}" for p in prompts]
        cal_prompts = [p + " {}" for p in cal_prompts]
    return prompts, cal_prompts

# read in the prompt template to a string
def load_prompt_template(file_path):

    with open(file_path, "r") as file:
        prompt_template = file.read()

    return prompt_template

def load_sentence_pairs(file_path):
    # Initialize an empty list to store the tuples
    data = []
    with open(file_path, 'r', encoding="utf-8") as file:
    # Iterate over each line in the file
        for line in file:
            # Split the line into fields using tab (\t) as the delimiter
            fields = line.strip().split('\t')
            # Create a tuple from the fields and append it to the data list
            data.append(tuple(fields))

    return data

def random_sample(input_list):
    # Use random.choice to select one element from the list
    sampled_element = random.choice(input_list)

    return sampled_element

def format_example(pair_tuple):
    return f"AAE: {pair_tuple[0]}\tSAE: {pair_tuple[1]}"

# Function to load attributes
def load_attributes(attribute_name, tok):
    with open(ATTRIBUTES_PATH.format(attribute_name), "r", encoding="utf8") as f:
        attributes = f.read().strip().split("\n")
    for a in attributes:
        assert len(tok.tokenize(" " + a)) == 1
    attributes = [tok.tokenize(" " + a)[0] for a in attributes]
    return attributes

# Function to compute probabilities for next/masked/sentinel token
def compute_probs(model, model_name, input_ids, labels):
    if model_name in GPT2_MODELS:
        output = model(input_ids=input_ids)
        probs = F.softmax(output.logits, dim=-1)[0][-1]
    elif model_name in ROBERTA_MODELS:
        output = model(input_ids=input_ids)
        probs = F.softmax(output.logits, dim=-1)[0][-2]
    elif model_name in T5_MODELS:
        output = model(input_ids=input_ids, labels=labels, decoder_input_ids=input_ids)
        probs = F.softmax(output.logits, dim=-1)[0][-1] 
    else:
        raise ValueError(f"Model {model_name} not supported.")
    return probs


# Function to retrieve attribute probabilities
def get_attribute_probs(prompt, attributes, model, model_name, tok, device, labels):
    input_ids = torch.tensor([tok.encode(prompt)])
    input_ids = input_ids.to(device)

    # Pass prompt through model
    probs = compute_probs(
        model, 
        model_name, 
        input_ids, 
        labels
    )

    # Select attribute probabilities
    probs_attribute = [
        probs[tok.convert_tokens_to_ids(a)].item() for a in attributes
    ]
    return probs_attribute

# Function to calibrate probabilities
def calibrate(probs, cal_probs, logprob=False):
    if logprob:
        return [(np.exp(p) - np.exp(cal_p)) for p, cal_p in zip(probs, cal_probs)]
    return [(p - cal_p) for p, cal_p in zip(probs, cal_probs)]

# Function to get the AAE and SAE sentences from the generated text
def extract_new_examples(text):
    # Modified regex to capture AAE sentences considering multiple sentence terminators and contexts
    aae_sentences = re.findall(r'AAE: (.*?)(?:\t|SAE:|\n)', text)
    # Modified regex to capture SAE sentences considering multiple sentence terminators and contexts
    sae_sentences = re.findall(r'SAE: (.*?)(?:\t|AAE:|\n|$)', text)

    return list(zip(aae_sentences, sae_sentences))

# Function to compute the loss for the generated dialogues
def dialogue_loss_function(aae_info, sae_info):
    aae_logits_d, aae_logits_nd = aae_info[3][:41], aae_info[3][41:]
    sae_logits_d, sae_logits_nd = sae_info[3][:41], sae_info[3][41:]

    # Convert lists to tensors
    aae_logits_d_tensor = torch.tensor(aae_logits_d)
    aae_logits_nd_tensor = torch.tensor(aae_logits_nd)
    sae_logits_d_tensor = torch.tensor(sae_logits_d)
    sae_logits_nd_tensor = torch.tensor(sae_logits_nd)

    # Calculate the probability ratios
    prob_ratio_d = sae_logits_d_tensor / aae_logits_d_tensor
    prob_ratio_nd = aae_logits_nd_tensor / sae_logits_nd_tensor

    # Calculate the combined loss
    combined_loss = - (torch.log(prob_ratio_d).mean() + torch.log(prob_ratio_nd).mean())
    
    # Calculate the mean of the combined loss
    return combined_loss

def write_dialogue_pairs_to_file(pairs_list, file_path):
    """
    Write a list of dialogue pairs to a file with pairs separated by tabs on the same line.

    Args:
        pairs_list (list): A list of dialogue pairs.
        file_path (str): The path to the file where the dialogue pairs will be written.
    """
    try:
        # Open the file in write mode
        with open(file_path, 'w') as file:
            # Write each dialogue pair to the file
            for pair in pairs_list:
                file.write(f"{pair[0]}\tT: {pair[1]}\n")  # Separating pairs with a tab

        print(f"Dialogue pairs written to {file_path} successfully.")

    except Exception as e:
        print(f"Error writing dialogue pairs to file: {e}")
