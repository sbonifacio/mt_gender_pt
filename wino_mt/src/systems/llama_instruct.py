from transformers import pipeline
import torch
import os
from tqdm import tqdm
import pdb

BATCH_SIZE = 50
CUDA_DEVICES="0"

LANG_CODE = {
   "pt": "Portuguese",
   "es": "Spanish",
   "fr": "French",
   "it": "Italian"   
}

def chunks(l, n):
    """
    Yield successive n-sized chunks from l.
    """
    for i in range(0, len(l), n):
        yield l[i:i + n]

def extract_translation(output):
    text = output[0]["generated_text"][-1]["content"]
    trans_sentence = text.split("\n")[0]
    return trans_sentence

def batch_translate_llama_instruct(lines, tgt_lang, src_lang = None):
    """
    Translate a list of sentences.
    Take care of batching.
    """
    # Load model
    model_name = "meta-llama/Llama-3.2-3B-Instruct"
    generator = pipeline(
        "text-generation", 
        model=model_name, 
        torch_dtype=torch.bfloat16,
        device = f"cuda:{CUDA_DEVICES}"
        #device_map="auto"
    )

    assert tgt_lang in LANG_CODE, f"{tgt_lang} is not supported"
    tgt_lang_code = LANG_CODE[tgt_lang]

    translations_dicts = []
    for chunk in tqdm(list(chunks(lines, BATCH_SIZE)), desc=f"size {BATCH_SIZE} chunks"):
        for sentence in chunk:
            message = [{"role": "user", "content": f"Translate the following text from English into {tgt_lang_code}. Return only the translation.\nEnglish: {sentence}\n{tgt_lang_code}:"}]
            output = generator(message, max_new_tokens=50,pad_token_id=generator.tokenizer.eos_token_id)
            out_dict = {"translatedText": extract_translation(output), "input": sentence}
            translations_dicts.append(out_dict)
       
    return translations_dicts


