
""" Usage:
    <file-name> --in=IN_FILE --src=SOURCE_LANGUAGE --tgt=TARGET_LANGUAGE --out=OUT_FILE [--debug]
"""
# External imports
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from peft import AutoPeftModelForSeq2SeqLM
import torch
import os
from tqdm import tqdm

BATCH_SIZE = 50 # Up to 128 should be fine?

def chunks(l, n):
    """
    Yield successive n-sized chunks from l.
    """
    for i in range(0, len(l), n):
        yield l[i:i + n]


def batch_translate_lora_nllb200(lines, tgt_lang, src_lang = None):
    """
    Translate a list of sentences.
    Take care of batching.
    """

    model_name = "../../fine-tuning/overlap_model"
    model = AutoPeftModelForSeq2SeqLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained("facebook/nllb-200-1.3B")

    tgt_lang="por_Latn"

    # Move the model to the GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {torch.cuda.device_count()} GPUs!")
    model.to(device)

    translations_dicts = []
    for chunk in tqdm(list(chunks(lines, BATCH_SIZE)), desc=f"size {BATCH_SIZE} chunks"):
        
        # Tokenize input text with source and target lang
        inputs = tokenizer(chunk, padding=True, truncation=True,return_tensors="pt").to(device)
        
        # Add language codes 
        lang_id = tokenizer.convert_tokens_to_ids(tgt_lang)
        
        # Generate translation
        outputs = model.generate(**inputs,forced_bos_token_id=lang_id)
        
        # Decode translated output
        translated_texts = tokenizer.batch_decode(outputs, skip_special_tokens=True)
            
        out_dicts = [{"translatedText": translated_text, "input": sent} for sent, translated_text in zip(chunk, translated_texts)]
        for out_dict in out_dicts:
            translations_dicts.append(out_dict)


    return translations_dicts


