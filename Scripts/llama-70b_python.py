import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, logging
from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig
import numpy as np
from torch import nn
from collections import defaultdict
import pandas as pd
import re
from pprint import pprint
from transformers import AutoTokenizer, AutoModelForCausalLM
import math

df = pd.read_csv('../Data/corpus_sentences.csv')

df = df[~df['Too weird?'].isin(['maybe', 'too weird', 'yes'])]

df = df.dropna(subset =['Sentence'])

binomial_alpha = df['Sentence (WordA and WordB)']
binomial_nonalpha = df['Sentence (WordB and WordA)']

#load in the 13b model

model_name_or_path = "meta-llama/Llama-2-13b-hf"
#model_basename = "model"

#use_triton = False

tokenizer = AutoTokenizer.from_pretrained(model_name_or_path) #we'll use chat-gpt2, but we could use another model if we wanted
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(model_name_or_path, return_dict_in_generate=True) #load the model
model.config.pad_token_id = model.config.eos_token_id
model.config.pad_token_id = model.config.eos_token_id
tokenizer.pad_token = tokenizer.eos_token


device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
#device = 'cpu'
model = model.to(device)

def to_tokens_and_logprobs(model, tokenizer, input_texts):
    input_ids = tokenizer(input_texts, padding=True, return_tensors="pt").input_ids
    input_ids = input_ids.to(device)
    
    outputs = model(input_ids)
    probs = torch.log_softmax(outputs.logits, dim=-1).detach()

    # collect the probability of the generated token -- probability at index 0 corresponds to the token at index 1
    probs = probs[:, :-1, :]
    input_ids = input_ids[:, 1:]
    gen_probs = torch.gather(probs, 2, input_ids[:, :, None]).squeeze(-1)

    batch = []
    for input_sentence, input_probs in zip(input_ids, gen_probs):
        text_sequence = []
        for token, p in zip(input_sentence, input_probs):
            if token not in tokenizer.all_special_ids:
                text_sequence.append((tokenizer.decode(token), p.item()))
        batch.append(text_sequence)
    return batch

#get alphabetical orderings
input_texts_alpha = binomial_alpha

n_batches = len(input_texts_alpha) / 50



input_texts_alpha = np.array_split(input_texts_alpha, n_batches)
input_texts_alpha = [x.tolist() for x in [*input_texts_alpha]]

batch_alpha = [[]]
timer = 0
for minibatch in input_texts_alpha:
  timer += 1
  print(timer)
  batch_placeholder = to_tokens_and_logprobs(model, tokenizer, minibatch)
  batch_alpha.extend(batch_placeholder)
  

batch_alpha = batch_alpha[1:]
sentence_probs_alpha = [sum(item[1] for item in inner_list[2:]) for inner_list in batch_alpha]

#get nonalphabetical orderings

input_texts_nonalpha = binomial_nonalpha

n_batches = len(input_texts_nonalpha) / 50
#input_texts_alpha = (np.array(np.array_split(input_texts_alpha, n_batches))).tolist() 



input_texts_nonalpha = np.array_split(input_texts_nonalpha, n_batches)
input_texts_nonalpha = [x.tolist() for x in [*input_texts_nonalpha]]



batch_nonalpha = [[]]
timer = 0
for minibatch in input_texts_nonalpha:
  timer += 1
  print(timer)
  batch_placeholder = to_tokens_and_logprobs(model, tokenizer, minibatch)
  batch_nonalpha.extend(batch_placeholder)
  

batch_nonalpha = batch_nonalpha[1:]

sentence_probs_nonalpha = [sum(item[1] for item in inner_list[2:]) for inner_list in batch_nonalpha]

#combine them into one dataframe

binom_probs = {}

for i,row in enumerate(df.itertuples()):
  binom = row[1] + ' and ' + row[2]
  binom_probs[binom] = [sentence_probs_alpha[i], sentence_probs_nonalpha[i]]


binom_probs_df = pd.DataFrame.from_dict(binom_probs, orient = 'index', columns = ['Alpha Probs', 'Nonalpha Probs'])
binom_probs_df.reset_index(inplace=True)
binom_probs_df.rename(columns = {'index': 'binom'}, inplace = True)

binom_probs_df['ProbAandB'] = binom_probs_df.apply(lambda row: math.exp(row['Alpha Probs']) / (math.exp(row['Alpha Probs']) + math.exp(row['Nonalpha Probs'])), axis=1)
#print(binom_probs_df)

binom_probs_df['log_odds'] = binom_probs_df['Alpha Probs'] - binom_probs_df['Nonalpha Probs']

#write into csv


binom_probs_df.to_csv('../Data/llama13b_unquantized_2afc_binom_ordering_prefs.csv', index = False)


