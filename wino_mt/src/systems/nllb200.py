from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
import os
from tqdm import tqdm

BATCH_SIZE = 50
CUDA_DEVICES = "0"

LANG_CODE = {
   "pt": "por_Latn",
   "es": "spa_Latn",
   "fr": "fra_Latn",
   "it": "ita_Latn"   
}

def chunks(l, n):
    """
    Yield successive n-sized chunks from l.
    """
    for i in range(0, len(l), n):
        yield l[i:i + n]


def batch_translate_nllb200(lines, tgt_lang, src_lang = None):
    """
    Translate a list of sentences.
    Take care of batching.
    """
    # Load the model and tokenizer
    model_name = "facebook/nllb-200-3.3B"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    # Retrieve target language code
    assert tgt_lang in LANG_CODE, f"{tgt_lang} is not supported"
    tgt_lang_code = LANG_CODE[tgt_lang]

    # Move the model to the GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = CUDA_DEVICES
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {torch.cuda.device_count()} GPUs!")
    model.to(device)

    translations_dicts = []
    for chunk in tqdm(list(chunks(lines, BATCH_SIZE)), desc=f"size {BATCH_SIZE} chunks"):
        
        # Tokenize input text 
        inputs = tokenizer(chunk, padding=True, truncation=True,return_tensors="pt").to(device)
        
        # Add language code
        lang_id = tokenizer.convert_tokens_to_ids(tgt_lang_code)
        
        # Generate translation
        outputs = model.generate(**inputs,forced_bos_token_id=lang_id)
        
        # Decode translated output
        translated_texts = tokenizer.batch_decode(outputs, skip_special_tokens=True)
            
        # Format translation results
        out_dicts = [{"translatedText": translated_text, "input": sent} for sent, translated_text in zip(chunk, translated_texts)]
        for out_dict in out_dicts:
            translations_dicts.append(out_dict)


    return translations_dicts

