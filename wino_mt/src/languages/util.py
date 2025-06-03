""" Usage:
    <file-name> --in=IN_FILE --out=OUT_FILE [--debug]
"""
# External imports
import logging
import spacy
import pdb
#import stanza
from pprint import pprint
from pprint import pformat
from docopt import docopt
from collections import defaultdict
from operator import itemgetter
from tqdm import tqdm
from enum import Enum
from spacy.tokens.token import Token
from typing import Dict

# Local imports

#=-----

class GENDER(Enum):
    """
    Enumerate possible genders.
    Ignore option resolves to words that should be ignored in particular language
    """
    male = 0
    female = 1
    neutral = 2
    unknown = 3
    ignore = 4 


SPACY_GENDER_TYPES = {
    "Masc": GENDER.male,
    "Fem": GENDER.female,
    "Neut": GENDER.neutral # seen in Dutch spacy
}

STANZA_GENDER_TYPES = {
    "Masc": GENDER.male,
    "Fem": GENDER.female,
    None: GENDER.neutral
}

# Winobias gender type conversion
WB_GENDER_TYPES = {
    "male": GENDER.male,
    "female": GENDER.female,
    "neutral": GENDER.neutral,
}

PYMORPH_GENDER_TYPES = {
    "masc": GENDER.male,
    "femn": GENDER.female,
    "neut": GENDER.neutral,
    None: GENDER.neutral
}

MORFEUSZ_GENDER_TYPES = {
    "m1": GENDER.male,
    "m2": GENDER.male,
    "m3": GENDER.male,
    "m1.m2": GENDER.male,
    "m1.m2.m3": GENDER.male,
    "f": GENDER.female,
    "m1.m2.m3.n": GENDER.neutral,
    "n": GENDER.neutral,
    None: GENDER.neutral
}

MORFEUSZ_GENDER_TAG_POSITION = 2


def get_morphology_dict(token: Token) -> Dict:
    """
    Parse a morphology dictionary from spacy token.
    @TODO: No better way to do this?
    """
    if "__" not in token.tag_:
        pdb.set_trace()
        return {}
        #raise AssertionError("No morphology support?")

    morphology = token.tag_.split("__")[1]

    if morphology == "_":
        return {}

    morphology_dict = dict([prop.split("=") for prop in morphology.split("|")])
    return morphology_dict

def get_morphology_dict_pt(token: Token) -> Dict:
    pos_tag = token.tag_  
    if "F" in pos_tag:
        gender = "Fem"
    elif "M" in pos_tag:
        gender = "Masc"
    else:
        return {}
    return {"Gender":gender}

def get_gender_from_token(token: Token):
    """
    Get gender indication from spacy token, if it exists
    """
    # Weird spacy bug? "au" should be male
    if (token.lang_ == "fr") and (token.text == "au") and (token.tag_.startswith("DET")):
        return GENDER.male

    # Italian spacy doesn't seem to split correctly
    if (token.lang_ == "it") and (token.text.startswith("dell'")):
        return GENDER.male

    if (token.lang_=="pt"):
        morph_dict = get_morphology_dict_pt(token)
    else:
        morph_dict = get_morphology_dict(token)
    if "Gender" not in morph_dict:
        return None

    morph_gender = SPACY_GENDER_TYPES[morph_dict["Gender"]]
    return morph_gender

def get_gender_from_token_stanza(token):
    """
    Get gender indication from spacy token, if it exists
    """
    if not token.feats: return None
    feats = token.feats.split("|") #"[Mood=Ind|Number=Sing|Person=3|Tense=Past|VerbForm=Fin]"
    feats_dict = {feat.split('=')[0]: feat.split('=')[1] for feat in feats}

    if "Gender" not in feats_dict: return None
    gender = STANZA_GENDER_TYPES[feats_dict["Gender"]]
    return gender



if __name__ == "__main__":
    # Parse command line arguments
    args = docopt(__doc__)
    inp_fn = args["--in"]
    out_fn = args["--out"]
    debug = args["--debug"]
    if debug:
        logging.basicConfig(level = logging.DEBUG)
    else:
        logging.basicConfig(level = logging.INFO)

    nlp = spacy.load("es")
    doc = nlp('Las naranjas y las manzanas se parecen')
    logging.info(list(map(get_morphology_dict, doc)))

    logging.info("DONE")
