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
from torch import cuda, bfloat16

df = pd.read_csv('corpus_sentences.csv')

df = df[~df['Too weird?'].isin(['maybe', 'too weird', 'yes'])]

df = df.dropna(subset =['Sentence'])
# Create new columns 'AandB' and 'BandA'
df['AandB'] = 'Next item: ' + df['WordA'] + ' and ' + df['WordB']
df['BandA'] = 'Next item: ' + df['WordB'] + ' and ' + df['WordA']

# Extracting columns for analysis
binomial_alpha = df['AandB']
binomial_nonalpha = df['BandA']

##########################################################################################################################################################
#70b version
###########################################################################################################################################################

#load in the 70b model

from torch import cuda, bfloat16
import transformers

model_id = 'meta-llama/Meta-Llama-3-70B'

device = f'cuda:{cuda.current_device()}' if cuda.is_available() else 'cpu'

# set quantization configuration to load large model with less GPU memory
# this requires the `bitsandbytes` library
bnb_config = transformers.BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type='nf4',
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=bfloat16
)

# begin initializing HF items, need auth token for these
model_config = transformers.AutoConfig.from_pretrained(
    model_id,
)

model = transformers.AutoModelForCausalLM.from_pretrained(
    model_id,
    trust_remote_code=True,
    config=model_config,
    quantization_config=bnb_config,
    device_map='auto',
)

model.config.pad_token_id = model.config.eos_token_id

model.eval()
print(f"Model loaded on {device}")

tokenizer = transformers.AutoTokenizer.from_pretrained(
    model_id,
)

tokenizer.pad_token = tokenizer.eos_token

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

n_batches = 20



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

n_batches = 20
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


binom_probs_df.to_csv('llama3_70b_2afc_just_binom_ordering_prefs.csv', index = False)