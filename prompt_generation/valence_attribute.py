import nltk
from nltk import pos_tag
from transformers import T5Tokenizer, GPT2Tokenizer

# Download NLTK resources
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

# tok = T5Tokenizer.from_pretrained("t5-small")
tok = GPT2Tokenizer.from_pretrained("gpt2-large")

def get_adjectives(lis):
  adjs = []

  # Check each word
  for word in lis:

    # Use POS tagging to get the part-of-speech
    tags = pos_tag([word])
    if tags[0][1] == "JJ":  # Check if the tag is "JJ" (adjective)
      adjs.append(word)

  return adjs

if __name__ == "__main__":
  
  valence_words = []
  scores = []

  with open('/content/hidden_bias_dialects/data/NRC-VAD-Lexicon.txt') as f:
    for i in f.readlines():
      line = i.rstrip('\n')
      word, score, _, _ = line.split('\t')
      valence_words.append(word)
      scores.append(float(score))

  high_valence = [valence_words[i] for i in range(len(scores)) if scores[i] > 0.67]
  low_valence = [valence_words[i] for i in range(len(scores)) if scores[i] < 0.33]

  with open('/content/hidden_bias_dialects/data/attributes/valence_lexicon.txt', 'w') as f2:
    for i in high_valence:
      f2.write(i + '\n')
    for i in low_valence:
      f2.write(i + '\n')

  high_valence_adjs = get_adjectives(high_valence)
  low_valence_adjs = get_adjectives(low_valence)

  high_valence_one_token = [i for i in high_valence_adjs if len(tok.tokenize(" " + i)) == 1]
  low_valence_one_token = [i for i in low_valence_adjs if len(tok.tokenize(" " + i)) == 1]

  with open('/content/hidden_bias_dialects/data/attributes/valence_gpt2.txt', 'w') as f3:
    for i in high_valence_one_token:
      f3.write(i + '\n')
    for i in low_valence_one_token:
      f3.write(i + '\n')

  print(f'There are {len(high_valence_one_token)} high valence one-token adjectives')
  print(f'There are {len(low_valence_one_token)} high valence one-token adjectives')