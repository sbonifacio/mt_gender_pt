""" Usage:
    <file-name> --in=IN_FILE --out=OUT_FILE [--debug]
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
from collections import Counter
#import stanza

# Local imports
from languages.util import GENDER, get_gender_from_token_stanza
#=-----

class StanzaPredictor:
    """
    Class for stanza supported languages.
    """
    def __init__(self, lang: str):
        """
        Init spacy for the specified language code.
        """
        assert lang in ["pt"]
        self.lang = lang
        self.cache = {}    # Store calculated professions genders

        stanza.download("pt")
        self.nlp = stanza.Pipeline("pt")
        #self.nlp = spacy.load(self.lang, disable = ["parser", "ner"])


    def get_gender(self, profession: str, translated_sent = None, entity_index = None, ds_entry = None) -> GENDER:
        """
        Predict gender of an input profession.
        """
        if profession not in self.cache:
            self.cache[profession] = self._get_gender(profession)

        return self.cache[profession]

    def _get_gender(self, profession: str) -> GENDER:
        """
        Predict gender, without using cache
        """
        if not profession.strip():
            # Empty string
            return GENDER.unknown

        doc = self.nlp(profession)
        
        observed_genders = [get_gender_from_token_stanza(token) for sent in doc.sentences for token in sent.tokens if token]
        observed_genders_filtered = [gender for gender in observed_genders if gender is not None]

        #observed_genders = [gender for gender in map(get_gender_from_token_stanza, toks)
        #                    if gender is not None]

        if not observed_genders_filtered:
            # No observed gendered words - return unknown
            return GENDER.unknown

        # Return the most commonly observed gender
        return Counter(observed_genders_filtered).most_common()[0][0]

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

    logging.info("DONE")
