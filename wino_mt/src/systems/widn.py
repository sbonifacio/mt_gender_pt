import pdb
import os, requests
import logging

LANG_CODE = {
   "pt": "pt-pT",
}

def widn_translate(sents, target_language, source_language = None):
    """
    Run widn on a batch of sentences.
    """

    # Retrieve target language code
    assert target_language in LANG_CODE, f"{target_language} is not supported"
    tgt_lang_code = LANG_CODE[target_language]

    # Checks to see if the subscription key is available as an environment variable
    if 'WIDN_SECRET_TOKEN' in os.environ:
        subscriptionKey = os.environ['WIDN_SECRET_TOKEN']
    else:
        logging.error('Environment variable for WIDN_SECRET_TOKEN is not set.')
        raise ValueError
    

    # Prepare request
    url = "https://api.widn.ai/v1/translate"
    headers = {
    "Content-Type": "application/json",
    "X-Api-Key": subscriptionKey
    }

    model_name="sugarloaf"
    body = {
        "sourceText": sents,
        "config": {
            "model": model_name,
            "sourceLocale": "en",
            "targetLocale": tgt_lang_code
        }
    }

    # Post request
    request = requests.post(url, headers=headers, json=body)
    response = request.json()["targetText"]

    if (len(response) != len(sents)):
        pdb.set_trace()
        raise AssertionError

    # Format translation results
    trans = []
    for (cur_resp, sent) in zip(response, sents):
        trans.append({"translatedText": cur_resp,
                      "input": sent})
    return trans

