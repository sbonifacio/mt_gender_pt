
from transformers import pipeline
import torch
import os
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
import pdb

BATCH_SIZE = 50
CUDA_DEVICES = "0"

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
    """
    Extract translation result from output
    """
    text = output[0]["generated_text"]
    translation = text.split("\n ")[-1]
    return translation

def batch_translate_tower_instruct(lines, tgt_lang, src_lang = None):
    """
    Translate a list of sentences.
    Take care of batching.
    """
    # Load the model
    model_name = "Unbabel/TowerInstruct-7B-v0.2"

    # Retrieve target language code
    assert tgt_lang in LANG_CODE, f"{tgt_lang} is not supported"
    tgt_lang_code = LANG_CODE[tgt_lang]

    pipe = pipeline("text-generation", 
                    model=model_name, 
                    torch_dtype=torch.bfloat16, 
                    device=f"cuda:{CUDA_DEVICES}")

    translations_dicts = []
    for chunk in tqdm(list(chunks(lines, BATCH_SIZE)), desc=f"size {BATCH_SIZE} chunks"):
        for sentence in chunk:
            
            # Prepare input prompt
            message = [{"role": "user", "content": f"Translate the following text from English into {tgt_lang_code}.\nEnglish: {sentence}\n{tgt_lang_code}:"}]
            prompt = pipe.tokenizer.apply_chat_template(message, tokenize=False, add_generation_prompt=True)
            
            # Generate translation
            output = pipe(prompt, max_new_tokens=256, do_sample=False)

            # Format translation results
            out_dict = {"translatedText": extract_translation(output), "input": sentence}
            translations_dicts.append(out_dict)

    return translations_dicts



def batch_translate_tower(lines, tgt_lang, src_lang = None):
    """
    Translate a list of sentences.
    Take care of batching.
    """
    # Load the model and tokenizer´
    model_name = "Unbabel/TowerBase-7B-v0.1"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)

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
        for sentence in chunk:

            # Prepare input prompt
            prompt = f"English: {sentence}\n{tgt_lang_code}:"

            # Tokenize the input text
            inputs = tokenizer(prompt, return_tensors="pt").to(device)
            
            # Generate translation
            outputs = model.generate(**inputs, max_new_tokens=50,pad_token_id=tokenizer.eos_token_id)
            
            # Decode the translated output
            translated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

            # Format translation results
            out_dict = {"translatedText": translated_text.split(f"{tgt_lang_code}: ")[1], "input": sentence}
            translations_dicts.append(out_dict)

    return translations_dicts

