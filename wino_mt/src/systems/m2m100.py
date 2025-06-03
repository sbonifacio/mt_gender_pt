from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer
import torch
import os
from tqdm import tqdm

BATCH_SIZE = 50 
CUDA_DEVICES = "0"

def chunks(l, n):
    """
    Yield successive n-sized chunks from l.
    """
    for i in range(0, len(l), n):
        yield l[i:i + n]


def batch_translate_m2m100(lines, tgt_lang, src_lang = None):
    """
    Translate a list of sentences.
    Take care of batching.
    """
    # Load the model and tokenizer
    model_name = "facebook/m2m100_1.2B"
    tokenizer = M2M100Tokenizer.from_pretrained(model_name)
    model = M2M100ForConditionalGeneration.from_pretrained(model_name)

    # Move the model to the GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = CUDA_DEVICES
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {torch.cuda.device_count()} GPUs!")
    model.to(device)

    translations_dicts = []
    for chunk in tqdm(list(chunks(lines, BATCH_SIZE)), desc=f"size {BATCH_SIZE} chunks"):
        
        # Tokenize input text 
        tokenizer.src_lang = "en"
        inputs = tokenizer(chunk, padding=True, truncation=True,return_tensors="pt").to(device)
        
        # Add language codes
        lang_id = tokenizer.get_lang_id(tgt_lang)

        # Generate translation
        outputs = model.generate(**inputs,forced_bos_token_id=lang_id)
        
        # Decode the translated output
        translated_texts = tokenizer.batch_decode(outputs, skip_special_tokens=True)
            
        # Format translation results
        out_dicts = [{"translatedText": translated_text, "input": sent} for sent, translated_text in zip(chunk, translated_texts)]
        for out_dict in out_dicts:
            translations_dicts.append(out_dict)


    return translations_dicts
