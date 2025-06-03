
""" Usage:
    <file-name> --in=IN_FILE --src=SOURCE_LANGUAGE --tgt=TARGET_LANGUAGE --out=OUT_FILE [--debug]
"""
# External imports
from transformers import pipeline
import torch
import os
from tqdm import tqdm

BATCH_SIZE = 50 # Up to 128 should be fine?

LANG_CODE = {
   "pt": "Portuguese",
   "es": "Spanish",
   "fr": "French",
   "it": "Italian"   
}

def chunks_llama(l, n):
    """
    Yield successive n-sized chunks from l.
    """
    for i in range(0, len(l), n):
        yield l[i:i + n]

def extract_translation(output,tgt_lang_code):
    trans_sentence = output[0]["generated_text"]
    return trans_sentence.split(f"{tgt_lang_code}: ")[1].split("\n")[0]

def batch_translate_llama(lines, tgt_lang, src_lang = None):
    """
    Translate a list of sentences.
    Take care of batching.
    """
    # Load the model and tokenizer
    model_name = "meta-llama/Llama-3.2-3B"
    generator = pipeline(
        "text-generation", 
        model=model_name, 
        torch_dtype=torch.bfloat16,
        device = "cuda:0"
        #device_map="auto"
    )

    assert tgt_lang in LANG_CODE, f"{tgt_lang} is not supported"
    tgt_lang_code = LANG_CODE[tgt_lang]

    translations_dicts = []
    for chunk in tqdm(list(chunks_llama(lines, BATCH_SIZE)), desc=f"size {BATCH_SIZE} chunks"):
        
        # Create prompt
        prompts = [f"Translate the following text from English into {tgt_lang_code}.\nEnglish: {sentence}\n{tgt_lang_code}:" for sentence in chunk]
        
        # Generate translation
        outputs = generator(prompts, max_new_tokens=50,pad_token_id=generator.tokenizer.eos_token_id)
    

        out_dicts = [{"translatedText": extract_translation(translated_text,tgt_lang_code), "input": sent} for sent, translated_text in zip(chunk, outputs)]
        for out_dict in out_dicts:
            translations_dicts.append(out_dict)


    return translations_dicts


def llama(sents, target_language, source_language):
    # Load the model and tokenizer
    model_name = "meta-llama/Llama-3.2-1B"
    generator = pipeline(
        "text-generation", 
        model=model_name, 
        torch_dtype=torch.bfloat16,
        device = "cuda:0"
        #device_map="auto"
    )

    prompts = ["Translate the following text from English into Portuguese.\nEnglish: "+sentence+"\nPortuguese:" for sentence in sents]
        
    # Generate translation
    outputs = generator(prompts, max_new_tokens=50,pad_token_id=generator.tokenizer.eos_token_id)

    trans = [{"translatedText": extract_translation(translated_text), "input": sent} for sent, translated_text in zip(sents, outputs)]
   
    return trans
