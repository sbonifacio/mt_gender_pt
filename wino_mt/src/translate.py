""" Usage:
    <file-name> --trans=TRANSLATION_SERVICE --in=IN_FILE --src=SOURCE_LANGUAGE --tgt=TARGET_LANGUAGE --out=OUT_FILE [--debug]
"""
# External imports
import logging
import pdb
from pprint import pprint
from pprint import pformat
from docopt import docopt
from collections import defaultdict
from operator import itemgetter
from tqdm import tqdm
from google.cloud import translate
import html
import time

# Local imports
from systems.bing_translate import bing_translate
from systems.google_translate import google_translate
from systems.amazon_translate import aws_translate
from systems.deepl_translate import deepl_translate
from systems.nllb200 import batch_translate_nllb200
from systems.m2m100 import batch_translate_m2m100
from systems.opusmt import batch_translate_opusmt
from systems.llama import batch_translate_llama
from systems.llama_instruct import batch_translate_llama_instruct
from systems.tower import batch_translate_tower, batch_translate_tower_instruct
from systems.eurollm import batch_translate_eurollm
from systems.widn import widn_translate
from systems.lora_nllb200 import batch_translate_lora_nllb200
#=-----

BATCH_SIZE = 50 # Up to 128 should be fine?

def chunks(l, n):
    """
    Yield successive n-sized chunks from l.
    """
    for i in range(0, len(l), n):
        yield l[i:i + n]

def batch_translate(trans_function, lines, tgt_lang, src_lang = None):
    """
    Translate a list of sentences.
    Take care of batching.
    """
    translations_dicts = []
    for chunk in tqdm(list(chunks(lines, BATCH_SIZE)), desc=f"size {BATCH_SIZE} chunks"):
        for out_dict in trans_function(chunk, tgt_lang, src_lang):
            translations_dicts.append(out_dict)
    return translations_dicts

TRANSLATION_SERVICE = {
   "google": google_translate,
   "nllb": batch_translate_nllb200,
   "m2m": batch_translate_m2m100,
   "llama": batch_translate_llama,
   "opus": batch_translate_opusmt,
   "llama-inst" : batch_translate_llama_instruct,
   "tower": batch_translate_tower,
   "tower-inst" : batch_translate_tower_instruct,
   "bing": bing_translate,
   "aws": aws_translate,
   "deepl": deepl_translate,
   "widn": widn_translate,
   "eurollm": batch_translate_eurollm,
   "lora-nllb":batch_translate_lora_nllb200
}

if __name__ == "__main__":
    # Parse command line arguments
    args = docopt(__doc__)
    trans_service = args["--trans"]
    inp_fn = args["--in"]
    src_lang = args["--src"]
    tgt_lang = args["--tgt"]
    out_fn = args["--out"]
    debug = args["--debug"]
    if debug:
        logging.basicConfig(level = logging.DEBUG)
    else:
        logging.basicConfig(level = logging.INFO)

    # Figure out the translation service to use
    assert trans_service in TRANSLATION_SERVICE, f"{trans_service} is not supported"
    trans_function = TRANSLATION_SERVICE[trans_service]

    lines = [line.strip() for line in open(inp_fn, encoding = "utf8")]

    if trans_service in ["nllb", "m2m", "opus","llama","llama-inst","tower","tower-inst","eurollm","lora-nllb"]:
        out_dicts = trans_function(lines, tgt_lang, src_lang)

    else:    
        out_dicts = batch_translate(trans_function, lines, tgt_lang, src_lang)

        
    with open(out_fn, "w", encoding = "utf8") as fout:
        for out_dict in out_dicts:
            fout.write("{} ||| {}\n".format(out_dict["input"],
                                            out_dict["translatedText"]))

    logging.info("DONE")
