import os, requests
import logging
import pdb
import json

LANG_CODE = {
   "pt": "PT-PT",
   "es": "ES",
   "fr": "FR",
   "it": "IT"   
}

def deepl_translate(sents, target_language, source_language):
    """
    Run deepl on a batch of sentences.
    """


    auth_key="38cfe22d-db87-4b15-bba5-ab05d6876c61"
    if 'DEEPL_AUTH_KEY' in os.environ:
        auth_key = os.environ['DEEPL_AUTH_KEY']
    else:
        logging.error('Environment variable for DEEPL_AUTH_KEY is not set.')
        raise ValueError


    # Retrieve target language code
    assert target_language in LANG_CODE, f"{target_language} is not supported"
    tgt_lang_code = LANG_CODE[target_language]
    
 
    #base_url = 'https://api-free.deepl.com/v2/translate'
    base_url = 'https://api.deepl.com/v2/translate'
    batch_text = sents

    params = {
    'auth_key': auth_key,
    'text': batch_text,
    'target_lang': tgt_lang_code,  
    }
    
    request = requests.post(base_url, data=params).json()
    #response = request.json()
    if not ("translations" in request): 
        pdb.set_trace()
        raise AssertionError
    response = request["translations"]


    if (len(response) != len(sents)):
        pdb.set_trace()
        raise AssertionError

    trans = []
    for (cur_resp, sent) in zip(response, sents):
        trans.append({"translatedText": cur_resp["text"],
                      "input": sent})
    return trans
