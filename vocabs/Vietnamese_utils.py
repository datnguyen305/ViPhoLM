import re
import unicodedata

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
    codas = ['ng', 'nh', 'ch', 'u', 'n', 'o',
             'p', 'c', 'm', 'y', 'i', 't', 'k']

    if word in codas:
        return word

    return None


def split_phoneme(word: str) -> list[str, str, str]:
    onset, word = get_onset(word)

    medial, word = get_medial(word)

    nucleus, word = get_nucleus(word)

    coda = get_coda(word)

    return onset, medial, nucleus, coda


def is_Vietnamese(word: str) -> tuple[bool, tuple]:
    tone, word = get_tone(word)
    if not re.match(r"[a-zA-Zăâđưôơê]", word):
        return False, None

    # handling for special cases
    special_words_to_words = {
        "gin": "giin",     # gìn after being removed the tone
        "giêng": "giiêng",  # giếng after being removed the tone
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
            return False, None

        previousCharacterIsVowels = currentCharacterIsVowels

    # in case the word has the structure of a Vietnamese word, we check whether it satisfies the rule of phoneme combination
    onset, medial, nucleus, coda = split_phoneme(word)

    if nucleus is None:
        return False, None

    former_word = ""
    for component in [onset, medial, nucleus, coda]:
        if component is not None:
            former_word += component
    if former_word != word:
        return False, None

    if onset == "k" and medial is None and nucleus not in ["i", "y", "e", "ê", "iê", "yê", "ia", "ya"]:
        return False, None

    if onset == "c" and medial is None and nucleus in ["i", "y", "e", "ê", "iê", "yê", "ia", "ya"]:
        return False, None

    if onset == "q" and not medial == "u":
        return False, None

    if onset == "gh" and medial is None and nucleus not in ["i", "e", "ê", "iê"]:
        return False, None

    if onset == "g" and medial is None and nucleus in ["i", "e", "ê", "iê"]:
        return False, None

    if onset == "ngh" and medial is None and nucleus not in ["i", "e", "ê", "iê", "yê", "ia", "ya"]:
        return False, None

    if onset == "ng" and medial is None and nucleus in ["i", "e", "ê", "iê", "yê", "ia", "ya"]:
        return False, None

    if onset in ["r", "gi"] and medial is not None:
        return False, None

    if medial == "o" and nucleus not in ["a", "ă", "e"]:
        return False, None

    if medial == "u" and nucleus not in ['yê', 'ya', 'e', 'ê', 'y', 'ơ', "ô", 'a', 'â', 'ă']:
        return False, None

    if nucleus == "oo" and coda not in ["ng", "c"]:
        return False, None

    if nucleus == "ua" and coda is not None:
        return False, None

    if nucleus == "ia" and coda is not None:
        return False, None

    if nucleus == "ya" and coda is not None:
        return False, None

    if nucleus in ["ua", "uô"] and coda == "ph":
        return False, None

    if nucleus in ["yê", "iê"] and coda is None:
        return False, None

    if nucleus in ["ă", "â"] and coda is None:
        return False, None

    if medial == "o" and nucleus in ["iê", "yê", "ia", "ya"]:
        return False, None

    if medial is not None:
        if nucleus in ["u", "oo", "o", "ua", "uô", "ươ", "ưa", "ư"]:
            return False, None

        if nucleus in ["i", "e", "ê", "ia", "ya", "iê", "yê"] and coda in ["m", "ph"]:
            return False, None

    if coda == "o" and nucleus not in ["a", "e"]:
        return False, None

    if coda == "y" and nucleus not in ["a", "â"]:
        return False, None

    if coda == "i" and nucleus in ["ă", "â", "i", "e", "iê", "yê", "ia", "ya"]:
        return False, None

    if coda == "nh" and nucleus not in ["a", "i", "y", "ê"]:
        return False, None

    if coda == "ng" and nucleus not in ["a", "o", "ô", "u", "ư", "e", "iê", "ươ", "â", "ă", "uô", "oo"]:
        return False, None

    if coda == "ch" and nucleus not in ["i", "a", "ê", "y"]:
        return False, None

    if coda == "c" and nucleus in ["i", "ê", "e", "ơ"]:
        return False, None

    if nucleus == coda:
        return False, None

    return True, (onset, medial, nucleus, coda, tone)


# def compose_word(onset: str, medial: str, nucleus: str, coda: str, tone: str) -> str:
#     tone_map = {
#         '<`>': '\u0300',
#         '</>': '\u0301',
#         '<~>': '\u0303',
#         '<?>': '\u0309',
#         '<.>': '\u0323'
#     }
#     tone = tone_map[tone]

#     # process for the special case of medial + coda (hỏa, thủy, thuở, thỏa, ...)
#     # in this case, only "thuở" follows the general rule of tone marking, the others are the case that tones are marked on the medial.
#     if onset != "q" and medial is not None and nucleus is not None and coda is None and nucleus != "ơ":
#         medial += tone
#     else:
#         if coda is None:
#             nucleus = nucleus[0] + tone + nucleus[1:]
#         else:
#             nucleus = nucleus + tone

#     word = ""
#     if onset:
#         word += onset
#     if medial:
#         word += medial
#     if nucleus:
#         word += nucleus
#     if coda:
#         word += coda

#     if "gii" in word:
#         word = re.sub("gii", "gi", word)

#     return word

def compose_word(initial: str, rhyme: str, tone: str) -> tuple[str, None]:
    def get_glide(rhyme: str) -> str:
        if rhyme.startswith("w"):
            return "w"

        return None

    def get_final(rhyme: str) -> str:
        possible_finals = ['j', 'm', 'n', 'ŋ̟', 'ŋ', 'p', 'k', 't', 'k̟', 'w']
        for final in possible_finals:
            if rhyme.endswith(final):
                return final

        return None

    initial2_onset = {
        None: [None],
        "m": ["m"],
        "b": ["b"],
        "k": ["c", "k", "q"],
        "v": ["v"],
        "t": ["t"],
        "ʝ": ["d"],
        "d": ["đ"],
        "n": ["n"],
        "r": ["r"],
        "s": ["x"],
        "ʂ": ["s"],
        "l": ["l"],
        "h": ["h"],
        "f": ["ph"],
        "tʰ": ["th"],
        "ɣ": ["g", "gh"],
        "z": ["gi"],
        "t͡ɕ": ["ch"],
        "ʈ͡ʂ": ["tr"],
        "ɲ": ["nh"],
        "ŋ": ["ng", "ngh"],
        "χ": ["kh"],
    }

    glide2medial = {
        None: [None],
        "w": ["u", "o"],
    }

    vowel2nucleus = {
        None: [None],
        "iə": ["iê", "yê", "ia", "ya"],
        "uə": ["uô", "ua"],
        "ɯə": ["ươ", "ưa"],
        "a": ["a"],
        "ă": ["a", "ă"],
        "ə̆": ["â"],
        "i": ["i", "y"],
        "ɛ": ["e"],
        "e": ["ê"],
        "u": ["u"],
        "ɯ": ["ư"],
        "ɔ": ["o"],
        "ɔː": ["oo"],
        "o": ["ô"],
        "ə": ["ơ"],
    }

    final2coda = {
        None: [None],
        "j": ["i", "y"],
        "m": ["m"],
        "n": ["n"],
        "ŋ̟": ["nh"],
        "ŋ": ["ng"],
        "p": ["p"],
        "t": ["t"],
        "k": ["c"],
        "k̟": ["ch"],
        "w": ["u", "o"],
    }

    glide: str = get_glide(rhyme)
    final: str = get_final(rhyme)
    vowel = rhyme
    if glide is not None:
        vowel = vowel.removeprefix(glide)
    if final is not None:
        vowel = vowel.removesuffix(final)

    coda_options = final2coda[final]
    if len(coda_options) == 1:
        coda = coda_options[0]
    else:
        # special cases where we have multiple options for the coda
        if coda_options == ["i", "y"]:
            if vowel in ["ă", "ə̆"]:
                coda = "y"
            else:
                coda = "i"

        if coda_options == ["u", "o"]:
            if vowel in ["ă", "ə̆", "e", "i", "iə", "ɯ", "ɯə"]:
                coda = "u"
            else:
                coda = "o"

    nucleus_options = vowel2nucleus[vowel]
    if len(nucleus_options) == 1:
        nucleus = nucleus_options[0]

    onset_options = initial2_onset[initial]
    if len(onset_options) == 1:
        onset = onset_options[0]

    # special cases where we have multiple options for the nucleus
    if nucleus_options == ["iê", "yê", "ia", "ya"]:
        nucleus = None
        if glide is None and final is None:
            nucleus = "ia"
        if glide is not None and final is None:
            nucleus = "ya"
        if glide is None and final is not None:
            nucleus = "iê"
        if glide is not None and final is not None:
            nucleus = "yê"
        if initial is None and glide is None:
            nucleus = "yê"
        # in the case where the syllabic components are connot correctly given
        if nucleus is None:
            return None

    if nucleus_options == ["uô", "ua"]:
        nucleus = None
        if final is None:
            nucleus = "ua"
        else:
            nucleus = "uô"
        # in the case where the syllabic components are connot correctly given
        if nucleus is None:
            return None

    if nucleus_options == ["ươ", "ưa"]:
        nucleus = None
        if final is None:
            nucleus = "ưa"
        else:
            nucleus = "ươ"
        # in the case where the syllabic components are connot correctly given
        if nucleus is None:
            return None

    if nucleus_options == ["i", "y"]:
        nucleus = None
        if initial is None and glide is None and final is None:
            nucleus = "y"
        elif glide is not None:
            nucleus = "y"
        else:
            nucleus = "i"
        # in the case where the syllabic components are connot correctly given
        if nucleus is None:
            return None

    if nucleus_options == ["a", "ă"]:
        if coda in ["u", "y"]:
            nucleus = "a"
        else:
            nucleus = "ă"

    if onset_options == ["c", "k", "q"]:
        if glide:
            onset = "q"
        else:
            if nucleus in ["i", "y", "e", "ê", "iê", "yê", "ia", "ya"]:
                onset = "k"
            else:
                onset = "c"

    if onset_options == ["ng", "ngh"]:
        if nucleus in ["i", "e", "ê", "iê", "yê", "ia", "ya"] and glide is None:
            onset = "ngh"
        else:
            onset = "ng"

    if onset_options == ["g", "gh"]:
        if nucleus in ["i", "e", "ê", "iê"]:
            onset = "gh"
        else:
            onset = "g"

    medial_options = glide2medial[glide]
    if len(medial_options) == 1:
        medial = medial_options[0]
    else:
        if onset == "q":
            medial = "u"
        else:
            # special cases where we have multiple options for the medial
            if nucleus in ["e", "a", "ă"]:
                medial = "o"
            else:
                medial = "u"

    tone_map = {
        "-": '',
        '˨˩': '\u0300',
        '˧˥': '\u0301',
        '˧ˀ˥': '\u0303',
        '˧˩': '\u0309',
        '˧ˀ˩': '\u0323'
    }
    tone = tone_map[tone]
    # process for the special case of medial + coda (hỏa, thủy, thuở, thỏa, ...)
    # in this case, only thuở, nhuệ/huệ/huề/duệ follows the general rule of tone marking, the others are the case that tones are marked on the medial.
    if onset != "q" and medial is not None and nucleus is not None and coda is None and nucleus not in ["ơ", "ê"]:
        medial += tone
    else:
        if coda is None:
            nucleus = nucleus[0] + tone + nucleus[1:]
        else:
            nucleus = nucleus + tone

    word = ""
    if onset:
        word += onset
    if medial:
        word += medial
    if nucleus:
        word += nucleus
    if coda:
        word += coda

    if "gii" in word:
        word = re.sub("gii", "gi", word)

    word = unicodedata.normalize('NFC', word)

    return word

def convert_Vietnamese_to_IPA(syllable: str) -> list[str]:
    assert re.search(r"\s+", syllable) is None, "The input must be a syllable"

    onset2ipa = {
        None: None,
        "m": "m",
        "b": "b",
        "c": "k",
        "k": "k",
        "v": "v",
        "t": "t",
        "d": "ʝ",
        "đ": "d",
        "n": "n",
        "r": "r",
        "x": "s",
        "s": "ʂ",
        "l": "l",
        "h": "h",
        "ph": "f",
        "th": "tʰ",
        "g": "ɣ",
        "gh": "ɣ",
        "gi": "z",
        "ch": "t͡ɕ",
        "tr": "ʈ͡ʂ",
        "nh": "ɲ",
        "ng": "ŋ",
        "ngh": "ŋ",
        "kh": "χ",
        "q": "k",
        # for foreign characters
        "w": "w",
        "f": "f",
        "z": "z",
        "j": "j",
        "p": "p"
    }

    medial2ipa = {
        None: None,
        # "u": "u̯",
        # "o": "u̯"
        "u": "w",
        "o": "w"
    }

    nucleus2ipa = {
        None: None,
        "iê": "iə",
        "yê": "iə",
        "ia": "iə",
        "ya": "iə",
        "uô": "uə",
        "ua": "uə",
        "ươ": "ɯə",
        "ưa": "ɯə",
        "a": "a",
        "ă": "ă",
        "â": "ə̆",
        "i": "i",
        "y": "i",
        "e": "ɛ",
        "ê": "e",
        "u": "u",
        "ư": "ɯ",
        "o": "ɔ",
        "oo": "ɔː",
        "ô": "o",
        "ơ": "ə"
    }

    coda2ipa = {
        None: None,
        "i": "j",
        "y": "j",
        "m": "m",
        "n": "n",
        "nh": "ŋ̟",
        "ng": "ŋ",
        "p": "p",
        "t": "t",
        "c": "k",
        "k": "k",
        "ch": "k̟",
        "u": "w",
        "o": "w",
    }

    mark2ipa = {
        None: "-",
        "<`>": "˨˩",
        "<?>": "˧˩",
        "<~>": "˧ˀ˥",
        "</>": "˧˥",
        "<.>": "˧ˀ˩",
    }

    is_Vietnamese_word, components = is_Vietnamese(syllable)

    if not is_Vietnamese_word:
        return None

    onset, medial, nucleus, coda, mark = components

    initial = onset2ipa[onset]
    glide = medial2ipa[medial]
    vowel = nucleus2ipa[nucleus]
    final = coda2ipa[coda]
    tone = mark2ipa[mark]

    if vowel == "a" and final in ["j", "w"]:
        if coda in ["y", "u"]:
            vowel = "ă"

    return initial, glide, vowel, final, tone


def analyse_Vietnamese(syllable: str) -> tuple[str]:
    special_cases = {
        "ty": "ti",
        "tỳ": "tì",
        "hỷ": "hỉ",
        "lý": "lí",
        "kỳ": "kì",
        "ký": "kí",
        "tỷ": "tỉ",
        "mỹ": "mĩ",
        "kỷ": "kỉ",
        "lỵ": "lị",
        "mỵ": "mị",
        "tý": "tí"
    }
    if syllable in special_cases:
        syllable = special_cases[syllable]

    components = convert_Vietnamese_to_IPA(syllable)

    if components:
        initial, medial, nucleus, final, tone = components

        rhyme = ""
        if medial:
            rhyme += medial
        if nucleus:
            rhyme += nucleus
        if final:
            rhyme += final

        return initial, rhyme, tone

    return None
