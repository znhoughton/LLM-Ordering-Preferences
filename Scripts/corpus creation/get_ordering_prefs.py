#get ordering preference for binomials

import pandas as pd
import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import numpy as np
from torch import nn
import re
from pprint import pprint
import os
import shutil
from transformers import AutoModelForCausalLM, TRANSFORMERS_CACHE
#python get_ordering_prefs.py --model 'allenai/OLMo-7B-0424-hf' --wordlist '../Data/nonce_binoms.csv' --checkpoint 'step500-tokens2B'
#user supplies two arguments:
#huggingface model
#csv file with list of words, named Word1 and Word2 respectively, where Word1 is the alphabetically first word in the binomial

def my_full_path(string):
    script_dir = os.path.dirname(__file__)
    return os.path.normpath(os.path.join(script_dir, string))

parser = argparse.ArgumentParser()

parser.add_argument('--model', type=str)
parser.add_argument('--wordlist', type=my_full_path)
parser.add_argument('--checkpoint', type=str) #optional checkpoint for Olmo Model
parser.add_argument('--print_checkpoints', action='store_true')


args = parser.parse_args()

model_name = args.model 
wordlist = args.wordlist

##functions

def to_tokens_and_logprobs(model, tokenizer, input_texts):
  
    input_ids = tokenizer(input_texts, padding=True, return_tensors="pt").to(device).input_ids
  
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

def get_probs(ordering):
    if ordering == 'alpha':
        order = 'AandB'
    if ordering == 'nonalpha':
        order = 'BandA'
    input_texts = data_just_binoms[order]
    n_batches = 10
    #print(n_batches)

    input_texts = np.array_split(input_texts, n_batches)
    input_texts = [x.tolist() for x in [*input_texts]]

    batch = [[]]
    timer = 0
    for minibatch in input_texts:
        timer += 1
        print(timer)
        batch_placeholder = to_tokens_and_logprobs(model, tokenizer, minibatch)
        batch.extend(batch_placeholder)
        

    batch = batch[1:]
    sentence_probs = [sum(item[1] for item in inner_list[2:]) for inner_list in batch]
    return sentence_probs


def combine_results(sentence_probs_alpha, sentence_probs_nonalpha):
    binom_probs = {}
    for i, row in enumerate(data_just_binoms.iterrows()):
        index, data = row
        word1 = data['Word1']
        word2 = data['Word2']
        binom = word1 + ' and ' + word2
        binom_probs[binom] = [sentence_probs_alpha[i], sentence_probs_nonalpha[i]]

    binom_probs_df = pd.DataFrame.from_dict(binom_probs, orient = 'index', columns = ['Alpha Probs', 'Nonalpha Probs'])
    binom_probs_df.reset_index(inplace=True)
    binom_probs_df.rename(columns = {'index': 'binom'}, inplace = True)
    return binom_probs_df 


def main():
    sentence_probs_alpha = get_probs('alpha')
    sentence_probs_nonalpha = get_probs('nonalpha')
    binom_probs_df = combine_results(sentence_probs_alpha, sentence_probs_nonalpha)
    file_name = model_name.replace('/', '_')
    cache_dir = TRANSFORMERS_CACHE
    if args.checkpoint:
        path = f'../Data/{file_name}_{args.checkpoint}.csv'
        binom_probs_df.to_csv(path)
        model_dir = f"{cache_dir}/models--allenai--OLMo-7B-0424-hf/refs/{args.checkpoint}"
    else:
        path = f'../Data/{file_name}.csv'
        binom_probs_df.to_csv(path)
        model_dir = f"{cache_dir}/models--allenai--OLMo-7B-0424-hf/refs/main"

    # Delete the model files
    try:
        shutil.rmtree(model_dir, ignore_errors=True)
    except:
        print(f"Can't find {model_dir}")


import os
os.environ['HF_HOME'] = 'D:/'      

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

data_just_binoms = pd.read_csv(wordlist)

data_just_binoms['AandB'] = 'Next item: ' + data_just_binoms['Word1'] + ' and ' + data_just_binoms['Word2']
data_just_binoms['BandA'] = 'Next item: ' + data_just_binoms['Word2'] + ' and ' + data_just_binoms['Word1']
print(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, use_fast = True) 
tokenizer.pad_token = tokenizer.eos_token

if args.checkpoint:
    print(f"checkpoint {args.checkpoint} specified, loading from checkpoint")
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, revision = args.checkpoint, trust_remote_code = True)
else:
    print('no checkpoint specified, loading main model')
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16)

model.config.pad_token_id = model.config.eos_token_id


#device = 'cpu'
model = model.to(device)

main()

