import gc
import time
from collections import defaultdict
import os 
import numpy as np
import pandas as pd
import torch
from torch import cuda
from tqdm import tqdm  

from transformers import AutoModelForCausalLM, AutoTokenizer
from hf_olmo import OLMoForCausalLM, OLMoConfig, OLMoTokenizerFast


# ==========================================================
#  DEVICE / DTYPE SETUP
# ==========================================================
def get_device_and_dtype():
    if torch.cuda.is_available():
        device = "cuda"
        print("cuda is available")
        torch_dtype = torch.float16
    else:
        device = "cpu"
        torch_dtype = torch.float32
    return device, torch_dtype


# ==========================================================
#  MODEL / TOKENIZER LOADER
# ==========================================================
def load_model_and_tokenizer(model_name: str):
    print(f"\n🔧 Loading tokenizer + model for: {model_name}")
    t0 = time.time()

    device, torch_dtype = get_device_and_dtype()

    # --------- FIXED OLMO DETECTION ---------
    if "olmo-2" in model_name.lower():
        print("  • Detected OLMo-2 model — using AutoModel + AutoTokenizer (slow tokenizer)")
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,
            use_fast=False,
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch_dtype,
            trust_remote_code=True,
        )

    elif "olmo" in model_name.lower():
        print("  • Detected OLMo-1 model — using hf_olmo classes")
        config = OLMoConfig.from_pretrained(model_name)

        tokenizer = OLMoTokenizerFast.from_pretrained(model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        config.pad_token_id = tokenizer.pad_token_id

        model = OLMoForCausalLM.from_pretrained(
            model_name,
            config=config,
            torch_dtype=torch_dtype,
        )

    else:
        print("  • Loading HuggingFace model")
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            use_fast=True,
            trust_remote_code=True,
        )

        # ----------- FIX GPT-2 padding error here ----------- #
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        # ----------------------------------------------------- #

        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch_dtype,
            trust_remote_code=True,
        )

    model.to(device)
    model.eval()

    dt = time.time() - t0
    print(f"✅ Finished loading {model_name} in {dt:.1f} seconds.")

    return model, tokenizer, device




# ==========================================================
#  SEQUENCE LOG-PROBABILITY CALCULATION
# ==========================================================
@torch.no_grad()
def sequence_logprobs(model, tokenizer, texts, device, batch_size: int = 16):

    print(f"    → computing log-probs for {len(texts)} sequences")

    all_scores = []

    # Progress bar over batches
    for start in tqdm(range(0, len(texts), batch_size), desc="      batches"):
        batch_texts = texts[start:start + batch_size]

        enc = tokenizer(batch_texts, padding=True, return_tensors="pt")
        input_ids = enc["input_ids"].to(device)
        attention_mask = enc["attention_mask"].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits

        log_probs = torch.log_softmax(logits, dim=-1)
        log_probs = log_probs[:, :-1, :]
        target_ids = input_ids[:, 1:]
        target_mask = attention_mask[:, 1:]

        token_logprobs = log_probs.gather(2, target_ids.unsqueeze(-1)).squeeze(-1)
        token_logprobs = token_logprobs * target_mask

        seq_logprobs = token_logprobs.sum(dim=-1)
        all_scores.extend(seq_logprobs.cpu().tolist())

    return all_scores


# ==========================================================
#  PREFERENCE SCORING FOR BINOMIALS
# ==========================================================
def get_preferences(model, tokenizer, device,
                    df_binoms: pd.DataFrame,
                    prompt: str,
                    prompt_value: int,
                    model_tag: str,
                    batch_size: int = 16):

    print(f"\n🔍 Running preferences for prompt {prompt_value}: {repr(prompt)}")
    print(f"    {len(df_binoms)} binomials")

    prefix = prompt + " " if prompt and not prompt.endswith(" ") else prompt

    alpha_texts = (prefix + df_binoms["Word1"] + " and " + df_binoms["Word2"]).tolist()
    nonalpha_texts = (prefix + df_binoms["Word2"] + " and " + df_binoms["Word1"]).tolist()

    print("  • alpha ordering sequences")
    alpha_logprobs = sequence_logprobs(model, tokenizer, alpha_texts, device, batch_size)

    print("  • nonalpha ordering sequences")
    nonalpha_logprobs = sequence_logprobs(model, tokenizer, nonalpha_texts, device, batch_size)

    print("  • building result dataframe")
    out_df = pd.DataFrame({
        "WordA": df_binoms["Word1"],
        "WordB": df_binoms["Word2"],
        "binom": df_binoms["Word1"] + " and " + df_binoms["Word2"],
        "alpha_logprob": alpha_logprobs,
        "nonalpha_logprob": nonalpha_logprobs,
        "model": model_tag,
        "prompt_value": prompt_value,
        "prompt_text": prompt,
    })

    out_df["preference"] = out_df["alpha_logprob"] - out_df["nonalpha_logprob"]

    print("  ✔ done with this prompt")
    return out_df



# ==========================================================
#  MAIN SCRIPT (Crash-safe version)
# ==========================================================
if __name__ == "__main__":

    df_binoms = pd.read_csv("../../Data/nonce_and_attested_binoms.csv")
    print(f"📄 Loaded {len(df_binoms)} binomials from nonce_binoms.csv")

    list_of_models = {
        "gpt2xl":   "gpt2-xl",
        "olmo2_1b": "allenai/OLMo-2-0425-1B",
        "olmo7b":   "allenai/OLMo-7B-0424"
    }

    list_of_prompts = [
        "Next item: ", "example: ", "instance: ", "try this: ", " ",
        "People often talk about ",
        "The discussion centered around the ",
        "Experts frequently mention the ",
        "The topic today involves the ",
        "The article referred to the ",
        "The speaker mentioned the ",
        "The report highlighted the ",
        "Yesterday we talked about the ",
        "Recently there has been debate about the ",
        "The meeting addressed issues related to the ",
        "The presentation focused on ",
        "A common example involves the ",
        "One familiar pairing is the ",
    ]

    # ======================================================
    #  CRASH-SAFE OUTPUT FILE
    # ======================================================
    out_path = "../../Data/ALL_MODELS_ALL_PREFIXES_NOVE_AND_ATTESTED.csv"

    # If exists → load & resume
    if os.path.exists(out_path):
        print(f"🔁 Found existing results at {out_path}, loading...")
        final_df = pd.read_csv(out_path)
        print(f"   Loaded {len(final_df)} rows — will resume where we left off.")
    else:
        final_df = pd.DataFrame()
        print("✨ Starting fresh — no previous results found.")

    # Helper to check which (model, prompt) pairs are complete
    def already_done(model_tag, pval):
        if final_df.empty:
            return False
        return (
            (final_df["model"] == model_tag) &
            (final_df["prompt_value"] == pval)
        ).any()


    # ======================================================
    #  MAIN LOOPS
    # ======================================================
    for model_tag, model_name in list_of_models.items():

        print("\n\n=====================================================")
        print(f"🚀 MODEL: {model_tag} → {model_name}")
        print("=====================================================")

        model, tokenizer, device = load_model_and_tokenizer(model_name)

        for pval, prompt in enumerate(tqdm(list_of_prompts, desc=f"Prompts for {model_tag}")):

            # ----- SKIP IF ALREADY COMPLETED -----
            if already_done(model_tag, pval):
                print(f"⏩ Skipping {model_tag} — prompt {pval} (already done)")
                continue

            # ----- RUN COMPUTATION -----
            out_df = get_preferences(
                model=model,
                tokenizer=tokenizer,
                device=device,
                df_binoms=df_binoms,
                prompt=prompt,
                prompt_value=pval,
                model_tag=model_tag,
                batch_size=16,
            )

            # ----- APPEND TO FILE SAFELY -----
            if final_df.empty:
                out_df.to_csv(out_path, index=False)
            else:
                out_df.to_csv(out_path, mode="a", header=False, index=False)

            final_df = pd.concat([final_df, out_df], ignore_index=True)

            print(f"💾 Saved progress — total rows: {len(final_df)}")

        print(f"🗑 Clearing {model_tag} from memory")
        del model, tokenizer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

    print(f"\n✨ DONE! Saved {len(final_df)} rows to {out_path}")
