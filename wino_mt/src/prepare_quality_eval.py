import sys
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from peft import AutoPeftModelForSeq2SeqLM
import torch
import os
from tqdm import tqdm

BATCH_SIZE = 20 # Up to 128 should be fine?
#model_name = "../../fine-tuning/lora_nllb_model"
model_name = "facebook/nllb-200-1.3B"
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
#model = AutoPeftModelForSeq2SeqLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained("facebook/nllb-200-1.3B")

def chunks(l, n):
    """
    Yield successive n-sized chunks from l.
    """
    for i in range(0, len(l), n):
        yield l[i:i + n]


def batch_translate_lora_nllb200(lines, tgt_lang="por_Latn", src_lang = None):
    """
    Translate a list of sentences.
    Take care of batching.
    """

    

    # Move the model to the GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {torch.cuda.device_count()} GPUs!")
    model.to(device)

    final_translations = []
    for chunk in tqdm(list(chunks(lines, BATCH_SIZE)), desc=f"size {BATCH_SIZE} chunks"):
        
        # Tokenize the input text with the appropriate source and target languages
        inputs = tokenizer(chunk, padding=True, truncation=True,return_tensors="pt").to(device)
        
        # Add language codes to the inputs
        lang_id = tokenizer.convert_tokens_to_ids(tgt_lang)
        
        # Generate translation
        outputs = model.generate(**inputs,forced_bos_token_id=lang_id)
        
        # Decode the translated output
        translated_texts = tokenizer.batch_decode(outputs, skip_special_tokens=True)
            
        final_translations+=translated_texts
        

    return final_translations







if __name__ == "__main__":
    # Parse command line arguments
    #args = docopt(__doc__)
    #inp_fn = args["--in"]
    #out_fn = args["--out"]
    if len(sys.argv) > 1: 
        inp_fn = sys.argv[1]
        out_fn = sys.argv[2]
    else:
        raise AssertionError("please provide file paths")



    with open(inp_fn, 'r') as f1:
        lines = f1.readlines()

    translations = batch_translate_lora_nllb200(lines) 
    

    with open(out_fn, 'w') as f2:
        for line in translations:
            f2.write(line+"\n")

