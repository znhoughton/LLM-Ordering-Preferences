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
from hf_olmo import OLMoForCausalLM
#df = pd.read_csv('../../Data/nonce_binoms.csv')

#df = df[~df['Too weird?'].isin(['maybe', 'too weird', 'yes'])]

#df = df.dropna(subset =['Sentence'])
# Create new columns 'AandB' and 'BandA'
#prompt 2: 'example: '
#prompt 3: 'instance: '
#prompt 4: 'try this: '

from torch import cuda, bfloat16
import transformers

device = f'cuda:{cuda.current_device()}' if cuda.is_available() else 'cpu'

def load_model_and_tokenizer(model_name, torch_dtype=torch.float16):
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
        use_fast=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if "OLMo-7B" in model_name:
        model = OLMoForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch_dtype
        )
        model.config.pad_token_id = model.config.eos_token_id
        model = model.to(device)

    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch_dtype,
            trust_remote_code=True,
            device_map="auto"
        )
        if hasattr(model.config, "pad_token_id"):
            model.config.pad_token_id = model.config.eos_token_id

    model.eval()
    return model, tokenizer


def to_tokens_and_logprobs(model, tokenizer, input_texts):
    input_ids = tokenizer(input_texts, padding=True, return_tensors="pt").input_ids.to(device)
    outputs = model(input_ids)
    probs = torch.log_softmax(outputs.logits, dim=-1).detach()

    probs = probs[:, :-1, :]
    input_ids_shift = input_ids[:, 1:]

    gen_probs = torch.gather(probs, 2, input_ids_shift[:, :, None]).squeeze(-1)

    batch = []
    for ids, lp in zip(input_ids_shift, gen_probs):
        seq = []
        for tok, p in zip(ids, lp):
            if tok not in tokenizer.all_special_ids:
                seq.append((tokenizer.decode([tok]), p.item()))
        batch.append(seq)
    return batch


def get_preferences(model, tokenizer, prompt, prompt_value, model_tag):
    df = pd.read_csv('../../Data/nonce_binoms.csv')
    #df = df.dropna(subset=['Sentence'])

    df['AandB'] = prompt + df['Word1'] + ' and ' + df['Word2']
    df['BandA'] = prompt + df['Word2'] + ' and ' + df['Word1']

    binomial_alpha = df['AandB']
    binomial_nonalpha = df['BandA']

    # -------------------
    # Token logprobs (ALPHA)
    # -------------------
    n_batches = 20
    batches_alpha = np.array_split(binomial_alpha, n_batches)
    batches_alpha = [b.tolist() for b in batches_alpha]

    print(f"\n--- Running alphabetical orderings for {model_tag} ---")

    batch_alpha = []
    for i, minibatch in enumerate(batches_alpha, 1):
        print(f"Batch {i}/{n_batches}")
        out = to_tokens_and_logprobs(model, tokenizer, minibatch)
        batch_alpha.extend(out)

    sentence_probs_alpha = [
        sum(tok[1] for tok in seq[2:])
        for seq in batch_alpha
    ]

    # -------------------
    # Token logprobs (NONALPHA)
    # -------------------
    batches_nonalpha = np.array_split(binomial_nonalpha, n_batches)
    batches_nonalpha = [b.tolist() for b in batches_nonalpha]

    print(f"\n--- Running non-alphabetical orderings for {model_tag} ---")

    batch_nonalpha = []
    for i, minibatch in enumerate(batches_nonalpha, 1):
        print(f"Batch {i}/{n_batches}")
        out = to_tokens_and_logprobs(model, tokenizer, minibatch)
        batch_nonalpha.extend(out)

    sentence_probs_nonalpha = [
        sum(tok[1] for tok in seq[2:])
        for seq in batch_nonalpha
    ]

    # -------------------
    # Combine into DF
    # -------------------
    final = pd.DataFrame({
        "binom": df['Word1'] + " and " + df['Word2'],
        "alpha_prob": sentence_probs_alpha,
        "nonalpha_prob": sentence_probs_nonalpha
    })

    outname = f"{model_tag}_prompt{prompt_value+1}_validation.csv"
    final.to_csv(outname, index=False)
    print(f"Saved → {outname}\n")

#prompt 2: 'example: '
#prompt 3: 'instance: '
#prompt 4: 'try this: '
#prompt 5: " " (blank)
list_of_models = {
    #"olmo7b":  "allenai/OLMo-7B-0424",
    "olmo2_1b": "allenai/OLMo-2-0425-1B",
    "gpt2xl":  "gpt2-xl"
}

list_of_prompts = ['Next item: ', 'example: ', 'instance: ', 'try this: ', ' ']

import gc

for model_tag, model_name in list_of_models.items():
    torch.cuda.empty_cache()
    gc.collect()

    dtype = torch.float16 if "gpt2" not in model_name.lower() else torch.float32
    model, tokenizer = load_model_and_tokenizer(model_name, torch_dtype=dtype)

    for pval, prompt in enumerate(list_of_prompts):
        print(f"=== {model_tag.upper()} — Prompt {pval} ===")
        get_preferences(model, tokenizer, prompt, pval, model_tag)