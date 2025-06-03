
""" Usage:
    <file-name> --in=IN_FILE --src=SOURCE_LANGUAGE --tgt=TARGET_LANGUAGE --out=OUT_FILE [--debug]
"""
# External imports
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from tqdm import tqdm
import os

BATCH_SIZE = 50 
CUDA_DEVICES = "0"

LANG_CODE = {
   "pt": "Portuguese",
   "es": "Spanish",
   "fr": "French",
   "it": "Italian"   
}

def chunks_eurollm(l, n):
    """
    Yield successive n-sized chunks from l.
    """
    for i in range(0, len(l), n):
        yield l[i:i + n]


def batch_translate_eurollm(lines, tgt_lang, src_lang = None):
    """
    Translate a list of sentences.
    Take care of batching.
    """
    # Load the model and tokenizer
    model_id = "utter-project/EuroLLM-9B"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_id)

    assert tgt_lang in LANG_CODE, f"{tgt_lang} is not supported"
    tgt_lang_code = LANG_CODE[tgt_lang]

    # Move the model to the GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = CUDA_DEVICES
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {torch.cuda.device_count()} GPUs!")
    model.to(device)

    translations_dicts = []
    for chunk in tqdm(list(chunks_eurollm(lines, BATCH_SIZE)), desc=f"size {BATCH_SIZE} chunks"):
        
        # Create prompt
        prompts = [f"English: {sentence} {tgt_lang_code}: " for sentence in chunk]
        
        # Tokenize input text
        inputs = tokenizer(prompts, return_tensors="pt",padding=True).to(device)

        # Generate translations
        outputs = model.generate(**inputs, max_new_tokens=50,pad_token_id=tokenizer.eos_token_id)

        # Decode the translated output
        translated_texts= tokenizer.batch_decode(outputs, skip_special_tokens=True)

        # Format translation results
        out_dicts = [{"translatedText": translated_text.split(f"{tgt_lang_code}: ")[1], "input": sent} for sent, translated_text in zip(chunk, translated_texts)]
        for out_dict in out_dicts:
            translations_dicts.append(out_dict)


    return translations_dicts
