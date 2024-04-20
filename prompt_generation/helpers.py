import random
import os
import numpy as np
import torch
from torch.nn import functional as F
import re
import pandas as pd
from transformers import (
    GPT2LMHeadModel, 
    GPT2Tokenizer, 
    RobertaForMaskedLM, 
    RobertaTokenizer, 
    T5ForConditionalGeneration,
    T5Tokenizer,
    BertTokenizer, 
    BertModel
)

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from torch import nn

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

    # Calculate the semantic similarity between the sentences

    # Calculate the perplexity of the generated sentences

    # Calculate the combined loss
    combined_loss = - (torch.log(prob_ratio_d).mean() + torch.log(prob_ratio_nd).mean())
    
    # Calculate the mean of the combined loss
    return combined_loss

def append_dialogue_pairs_to_file(pairs_list, file_path):
    """
    Append a list of dialogue pairs to a file with pairs separated by tabs on the same line.

    Args:
        pairs_list (list): A list of dialogue pairs, where each pair is a tuple containing two strings.
        file_path (str): The path to the file where the dialogue pairs will be written.
    """
    try:
        # Open the file in append mode
        with open(file_path, 'a') as file:
            # Write each dialogue pair to the file
            for pair in pairs_list:
                file.write(f"{pair[0]}\t{pair[1]}\n")  # Separating pairs with a tab

    except Exception as e:
        print(f"Error appending dialogue pairs to file: {e}")

def get_most_effective_pairs(k, attack_set):
    df = pd.DataFrame(attack_set, columns=['dialogues', 'bias_loss'])
    df.drop_duplicates()
    # Min-max normalization
    min_value = df['bias_loss'].min()
    max_value = df['bias_loss'].max()

    # Apply min-max normalization
    df['normalized_loss'] = -1 + 2 * (df['bias_loss'] - min_value) / (max_value - min_value)

    #df['score'] = df['normalized_loss'] + df['sentence_similarity']

    # Sort the DataFrame by the 'Age' column in descending order
    df_sorted = df.sort_values(by='normalized_loss', ascending=True)
    top_k_rows = df_sorted.head(k)
    print(len(top_k_rows))

    return top_k_rows

# Function to normalize a column
def normalize_column(df, col_name):

    # Min-max normalization
    min_value = df['loss'].min()
    max_value = df['loss'].max()

    # Apply min-max normalization
    df['normalized_loss'] = -1 + 2 * (df['loss'] - min_value) / (max_value - min_value)

    #df['score'] = df['normalized_loss'] + df['sentence_similarity']

# Calculates cosine simlarity between two sentence embeddings
def cosine_similarity(sentence_embedding_1, sentence_embedding_2, debug = False):
  cos = nn.CosineSimilarity(dim=1, eps=1e-6)
  similarity = cos(sentence_embedding_1, sentence_embedding_2)
  if debug: print(f"Similarity between the two sentences: {similarity.item()}")

  return similarity.item()

# Computes the similarity score between two sentences
def similarity_score(sentence_1, sentence_2, debug = False):
  tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
  model = BertModel.from_pretrained('bert-base-uncased')

  # Tokenize the sentences
  tokens_1 = tokenizer.encode_plus(sentence_1, return_tensors='pt', padding=True)
  tokens_2 = tokenizer.encode_plus(sentence_2, return_tensors='pt', padding=True)

  # Obtain BERT embeddings
  with torch.no_grad():
      outputs_1 = model(**tokens_1)
      outputs_2 = model(**tokens_2)

  # Compute the mean of the token embeddings
  sentence_embedding_1 = torch.mean(outputs_1.last_hidden_state, dim=1)
  sentence_embedding_2 = torch.mean(outputs_2.last_hidden_state, dim=1)

  similarity = -cosine_similarity(sentence_embedding_1, sentence_embedding_2, debug)

  if debug: print(f"Similarity between sentences: {similarity:.4f}")

  return similarity