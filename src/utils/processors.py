import re
import numpy as np 


def clean_condition_text(text):

    text = text.lower()
    text = re.sub(r"[\n\r]", " ", text)
    text = re.sub(r"[^a-z0-9%.\- ]+", "", text)
    return text
