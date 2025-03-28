import unicodedata
import re

def get_tone(word: str):
    tone_map = {
        '\u0300': '<`>',
        '\u0301': '</>',
        '\u0303': '<~>',
        '\u0309': '<?>',
        '\u0323': '<.>',
    }
    decomposed_word = unicodedata.normalize('NFD', word)
    tone = None
    remaining_word = ''
    for char in decomposed_word:
        if char in tone_map:
            tone = tone_map[char]
        else:
            remaining_word += char
    remaining_word = unicodedata.normalize('NFC', remaining_word)
    
    return tone, remaining_word

def get_onset(word: str) -> tuple[str, str]:
    onsets = ['ngh', 'tr', 'th', 'ph', 'nh', 'ng', 'kh', 
              'gi', 'gh', 'ch', 'q', 'đ', 'x', 'v', 't', 
              's', 'r', 'n', 'm', 'l', 'k', 'h', 'g', 'd', 
              'c', 'b']
    
    # get the onset
    for onset in onsets:
        if word.startswith(onset):
            if onset != "q":
                word = word.removeprefix(onset)
            return onset, word

    return None, word

def get_medial(word: str) -> tuple[str, str]:
    O_MEDIAL = "o"
    U_MEDIAL = "u"

    if word.startswith("q"):
        # in Vietnamese, words starting with "q" always has "u" as the medial
        word = word.removeprefix("qu")
        return U_MEDIAL, word
    
    o_medial_cases = ["oa", "oă", "oe"]
    for o_medial_case in o_medial_cases:
        if word.startswith(o_medial_case):
            word = word.removeprefix("o")
            return O_MEDIAL, word
        
    if word.startswith("ua") or word.startswith("uô"):
        return None, word
    
    nucleuses = ['ê', 'y', 'ơ', 'a', 'â', 'ya']
    for nucleus in nucleuses:
        component = U_MEDIAL + nucleus
        if word.startswith(component):
            word = word.removeprefix("u")
            return U_MEDIAL, word
        
    return None, word

def get_nucleus(word: str) -> tuple[str, str]:
    nucleuses = ['oo', 'ươ', 'ưa', 'uô', 'ua', 'iê', 'yê', 
                 'ia', 'ya', 'e', 'ê', 'u', 'ư', 'ô', 'i', 
                 'y', 'o', 'ơ', 'â', 'a', 'o', 'ă']
    
    for nucleus in nucleuses:
        if word.startswith(nucleus):
            word = word.removeprefix(nucleus)
            return nucleus, word
        
    return None, word
    
def get_coda(word: str) -> str:
    codas = ['ng', 'nh', 'ch', 'u', 'n', 'o', 'p', 'c', 'm', 'y', 'i', 't']
    
    if word in codas:
        return word
    
    return None

def split_phoneme(word: str) -> list[str, str, str]:
    onset, word = get_onset(word)
    
    medial, word = get_medial(word)

    nucleus, word = get_nucleus(word)

    coda = get_coda(word)
    
    return onset, medial, nucleus, coda

def analyze_Vietnamese(word: str) -> tuple[bool, tuple]:
    tone, word = get_tone(word)
    if not re.match(r"[a-zA-Zăâđưôơê]", word):
        return None

    # handling for special cases
    special_words_to_words = {
        "gin": "giin",     # gìn after being removed the tone 
        "giêng": "giiêng", # giếng after being removed the tone
        "giêt": "giiêt",   # giết after being removed the tone
        "giêc": "giiêc",   # giếc (diếc) after being removed the tone
        "gi": "gii"        # gì after removing the tone 
    }

    if word in special_words_to_words:
        word = special_words_to_words[word]

    # check the total number of nucleus in word
    vowels = ['oo', 'ươ', 'ưa', 'uô', 'ua', 'iê', 'yê', 
              'ia', 'ya', 'e', 'ê', 'u', 'ư', 'ô', 'i', 
              'y', 'o', 'ơ', 'â', 'a', 'o', 'ă']
    currentCharacterIsVowels = False
    previousCharacterIsVowels = word[0] in vowels
    foundVowels = 0
    
    for character in word[1:]:
        if character in vowels:
            currentCharacterIsVowels = True
        else:
            currentCharacterIsVowels = False
        
        if currentCharacterIsVowels and not previousCharacterIsVowels:
            foundVowels += 1

        # in Vietnamese, each word has only one syllable    
        if foundVowels > 2:
            return None
            
        previousCharacterIsVowels = currentCharacterIsVowels
    
    # in case the word has the structure of a Vietnamese word, we check whether it satisfies the rule of phoneme combination
    onset, medial, nucleus, coda = split_phoneme(word)

    if nucleus is None:
        return None
    
    former_word = ""
    for component in [onset, medial, nucleus, coda]:
        if component is not None:
            former_word += component
    if former_word != word:
        return None
    
    if onset == "k" and medial is None and nucleus not in ["i", "y", "e", "ê", "iê", "yê", "ia", "ya"]:
        return None
    
    if onset == "c" and medial is None and nucleus in ["i", "y", "e", "ê", "iê", "yê", "ia", "ya"]:
        return None
    
    if onset == "q" and not medial == "u":
        return None
    
    if onset == "gh" and medial is None and nucleus not in ["i", "e", "ê", "iê"]:
        return None
    
    if onset == "g" and medial is None and nucleus in ["i", "e", "ê", "iê"]:
        return None
    
    if onset == "ngh" and medial is None and nucleus not in ["i", "e", "ê", "iê", "yê", "ia", "ya"]:
        return None
    
    if onset == "ng" and medial is None and nucleus in ["i", "e", "ê", "iê", "yê", "ia", "ya"]:
        return None
    
    if onset in ["r", "gi"] and medial is not None:
        return None
    
    if medial == "o" and nucleus not in ["a", "ă", "e"]:
        return None
    
    if medial == "u" and nucleus not in ['yê', 'ya', 'e', 'ê', 'y', 'ơ', "ô", 'a', 'â', 'ă']:
        return None
    
    if nucleus == "oo" and coda not in ["ng", "c"]:
        return None
    
    if nucleus == "ua" and coda is not None:
        return None
    
    if nucleus == "ia" and coda is not None:
        return None
    
    if nucleus == "ya" and coda is not None:
        return None
    
    if nucleus in ["ua", "uô"] and coda == "ph":
        return None
    
    if nucleus in ["yê", "iê"] and coda is None:
        return None
    
    if nucleus in ["ă", "â"] and coda is None:
        return None
    
    if medial == "o" and nucleus in ["iê", "yê", "ia", "ya"]:
        return None
    
    if medial is not None:
        if nucleus in ["u", "oo", "o", "ua", "uô", "ươ", "ưa", "ư"]:
            return None
        
        if nucleus in ["i", "e", "ê", "ia", "ya", "iê", "yê"] and coda in ["m", "ph"]:
            return None
        
    if coda == "o" and nucleus not in ["a", "e"]:
        return None
    
    if coda == "y" and nucleus not in ["a", "â"]:
        return None
    
    if coda == "i" and nucleus in ["ă", "â", "i", "e", "iê", "yê", "ia", "ya"]:
        return None
    
    if coda == "nh" and nucleus not in ["a", "i", "y", "ê"]:
        return None
    
    if coda == "ng" and nucleus not in ["a", "o", "ô", "u", "ư", "e", "iê", "ươ", "â", "ă", "uô", "oo"]:
        return None

    if coda == "ch" and nucleus not in ["i", "a", "ê", "y"]:
        return None

    if coda == "c" and nucleus in ["i", "ê", "e", "ơ"]:
        return None

    if nucleus == coda:
        return None

    onset, medial, nucleus, coda = assign_tone(onset, medial, nucleus, coda, tone)

    return (onset, medial, nucleus, coda)

def assign_tone(onset: str, medial: str, nucleus: str, coda: str, tone: str):
    tone_map = {
        None: "",
        '<`>': '\u0300',
        '</>': '\u0301',
        '<~>': '\u0303',
        '<?>': '\u0309',
        '<.>': '\u0323',
    }
    tone = tone_map[tone]

    if medial == "u" and nucleus == "ơ" and tone == "\u0309":
        nucleus += tone
        nucleus = unicodedata.normalize("NFC", nucleus)
        return onset, medial, nucleus, coda
    
    if onset == "gi" and nucleus == "i":
        onset += tone
        onset = unicodedata.normalize("NFC", onset)
        nucleus = None
        return onset, medial, nucleus, coda
    
    if onset == "gi" and nucleus == "iê":
        onset = onset[0]
        nucleus += tone
        nucleus = unicodedata.normalize("NFC", nucleus)
        return onset, medial, nucleus, coda
    
    if onset == "q":
        nucleus += tone
        nucleus = unicodedata.normalize("NFC", nucleus)
        return onset, medial, nucleus, coda
    
    if medial == "u" and nucleus == "ê":
        nucleus += tone
        nucleus = unicodedata.normalize("NFC", nucleus)
        return onset, medial, nucleus, coda
    
    if medial and coda is None:
        medial += tone
        medial = unicodedata.normalize("NFC", medial)

        return onset, medial, nucleus, coda
    
    if len(nucleus) == 1:
        nucleus += tone
    else:
        if coda:
            nucleus += tone
        else:
            nucleus = nucleus[0] + tone + nucleus[1]
    
    nucleus = unicodedata.normalize("NFC", nucleus)

    return onset, medial, nucleus, coda

def compose_word(onset: str, medial: str, nucleus: str, coda: str):
    if not (onset or medial or nucleus or coda):
        return None
    
    word = ""
    if onset:
        word += onset
    if medial:
        word += medial
    if nucleus:
        word += nucleus
    if coda:
        word += coda
        
    word = unicodedata.normalize("NFC", word)

    return word

