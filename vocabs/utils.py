import re
import unicodedata

def preprocess_sentence(sentence: str):
    sentence = sentence.lower()
    sentence = unicodedata.normalize("NFC", sentence)
    # remove all non-characters and punctuations
    sentence = re.sub(r"[-/“”!\*\&\$\.\?:;,\"'\(\[\]\(\)]", " ", sentence)
    sentence = " ".join(sentence.strip().split()) # remove duplicated spaces
    tokens = sentence.strip().split()
    
    return tokens
