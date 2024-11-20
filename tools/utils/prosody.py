import random
import re
from pyarabic.araby import strip_tashkeel

harakat = ["\u0650", "\u064e", "\u064f"]  # [kasra, fatha, damma, ]
sukun = ["\u0652"]  # [sukun]
mostly_saken = [
    "\u0627",
    "\u0648",
    "\u0649",
    "\u064a",
]  # [alef, waw, alef maqsurah, ya'a]
tnween_chars = [
    "\u064c",
    "\u064d",
    "\u064b",
]  # damm tanween, kasra tanween, fatha tanween, maddah
shadda_chars = ["\u0651"]
all_chars = list("إةابتثجحخدذرزسشصضطظعغفقكلمنهويىأءئؤ ")
prem_chars = harakat + sukun + mostly_saken + tnween_chars + shadda_chars + all_chars
CHANGE_LST = {
    "هذا": "هَاذَا",
    "هذه": "هَاذِه",
    "هذان": "هَاذَان",
    "هذين": "هَاذَين",
    "ذلك": "ذَالِك",
    "ذلكما": "ذَالِكُمَا",
    "ذلكم": "ذَالِكُم",
    "الله": "اللَّاه",
    "اللهم": "اللَّاهُمّ",
    "إله": "إِلَاه",
    "الإله": "الإِلَاه",
    "إلهي": "إِلَاهي",
    "إلهنا": "إِلَاهنا",
    "إلهكم": "إِلَاهكم",
    "إلههم": "إِلَاههم",
    "إلههن": "إِلَاههن",
    "رحمن": "رَحمَان",
    "طاوس": "طَاوُوس",
    "داود": "دَاوُود",
    "لكنه": "لَاكِنّهُ",
    "أولئك": "أُلَائِك",
    "أولئكم": "أُلَائِكُم",
}


def handle_space(plain_chars):
    if plain_chars[-1] == " ":
        return plain_chars[:-2]
    else:
        return plain_chars[:-1]


def remove_extra_harakat(pred):
    out = ""
    i = 0
    while i < len(pred):
        if i < len(pred) - 1:
            if pred[i] in harakat and pred[i + 1] in harakat:
                i += 1
                continue
        out += pred[i]
        i += 1
    return out


def extract_tf3eelav3(pred, verbose=False):
    pred = remove_extra_harakat(pred)
    chars = list(pred.replace("\u0622", "ءَا").strip())
    chars = [c for c in chars if c in prem_chars]
    chars = list(re.sub(" +", " ", "".join(chars).strip()))
    out = ""
    i = 0
    plain_chars = ""
    j = 0
    flag = True
    while i < len(chars) - 1 and flag:
        j += 1
        char = chars[i]
        if verbose:
            print(char)
        # plain_chars += char
        if char in all_chars:
            if char == " ":
                plain_chars += char
                i += 1
                continue
            # set up some vars
            next_char = chars[i + 1]
            if next_char == " ":
                next_char = chars[i + 2]
            if i < len(chars) - 2:
                next_next_char = chars[i + 2]
            if len(out) > 0:
                prev_char = out[-1]
            else:
                prev_char = ""
            # ----------------------
            if next_char in harakat:
                out += "1"
                plain_chars += char
            elif next_char in sukun:
                if prev_char != "0":
                    out += "0"
                    plain_chars += char
                elif (i + 1) == len(chars) - 1:
                    out = out[:-1] + "10"
                    plain_chars += char
                else:
                    plain_chars = handle_space(plain_chars) + char
            elif next_char in tnween_chars:
                if char != "ا":
                    plain_chars += char
                plain_chars += "ن"
                out += "10"
            elif next_char in shadda_chars:
                """added characters"""
                # 1
                if prev_char != "0":
                    plain_chars += char
                    plain_chars += char
                    out += "01"
                else:
                    plain_chars = handle_space(plain_chars) + char + char
                    out += "1"
                if i + 2 < len(chars):  # need to recheck this
                    if chars[i + 2] in harakat:  # in case shaddah not followed by harakah
                        i += 1
                    elif chars[i + 2] in tnween_chars:  # in case shaddah is followed by tanween
                        i += 1
                        # plain_chars += char
                        plain_chars += "ن"
                        # out += '10'
                        out += "0"
            elif next_char in all_chars:
                if prev_char != "0":
                    out += "0"
                    plain_chars += char
                elif prev_char == "0" and chars[i + 1] == " ":
                    out += "1"
                    plain_chars += char
                else:
                    plain_chars = handle_space(plain_chars) + char
                    # plain_chars += char
                i -= 1
            if next_next_char == " ":
                if char == "ه":
                    if next_char == harakat[0]:
                        plain_chars += "ي"
                        out += "0"
                    if next_char == harakat[2]:
                        plain_chars += "و"
                        out += "0"
            i += 2
        if j > 2 * len(chars):
            flag = False
            return None, None
            # raise Exception("error")

    if out[-1] != "0":
        out += "0"  # always add sukun to the end of baits if mutaharek
    if chars[-1] == harakat[0]:
        plain_chars += "ي"
    elif chars[-1] == tnween_chars[1]:
        plain_chars = plain_chars[:-1] + "ي"
    elif chars[-1] == harakat[1]:
        plain_chars += "ا"
    elif chars[-1] == harakat[2]:
        plain_chars += "و"
    elif chars[-1] == tnween_chars[0]:
        plain_chars = plain_chars[:-1] + "و"
    elif chars[-1] in "ىاوي" and chars[-2] not in tnween_chars:
        plain_chars += chars[-1]
    plain_chars_no_space = plain_chars.replace(" ", "")
    return plain_chars, out


def process_specials_before(bait):
    if bait[0] == "ا":
        bait = random.choice(["أَ", "إِ"]) + bait[1:]
    bait = bait.replace("وا ", "و ")
    if bait.find("وا") == len(bait) - 2:
        bait = bait[:-1]
    bait = bait.replace("وْا", "و")
    if bait.find("وْا") == len(bait) - 2:
        bait = bait[:-2] + "و"
    bait = bait.replace("الله", "اللاه")
    bait = bait.replace("اللّه", "الله")
    bait = bait.replace("إلَّا", "إِلّا")
    bait = bait.replace("نْ ال", "نَ ال")
    bait = bait.replace("لْ ال", "لِ ال")
    bait = bait.replace("إلَى", "إِلَى")
    bait = bait.replace("إذَا", "إِذَا")
    bait = bait.replace("ك ", "كَ ")
    bait = bait.replace(" ال ", " الْ ")
    bait = bait.replace("ْ ال", "ِ ال")

    out = []
    for word in bait.split(" "):
        cleaned_word = strip_tashkeel(word)
        for key in CHANGE_LST:
            if key in cleaned_word:
                cleaned_word = cleaned_word.replace(key, CHANGE_LST[key])
                out.append(cleaned_word)
                break
        else:
            out.append(word)

    bait = " ".join(out)

    if bait[1] in all_chars:
        bait = bait[0] + harakat[1] + bait[1:]
    out = []
    i = 0
    while i < len(bait):
        if bait[i] == "ا" and bait[i - 1] in tnween_chars:
            i += 1
            if i < len(bait) and bait[i] in harakat + sukun + tnween_chars + shadda_chars:
                """
                remove the case when we have any tashkeel comes after tnween chars
                """
                i += 1
            continue
        out += bait[i]
        i += 1

    return out


def process_specials_after(bait):
    bait = bait.replace("ةن", "تن")
    return bait


def get_verse_prosody(bait, verbose=False):
    bait = bait.strip()
    if len(bait) > 0:
        preprocessed = process_specials_before(bait)
        arudi_style, pattern = extract_tf3eelav3(preprocessed, verbose=verbose)
        if arudi_style and pattern:
            return process_specials_after(arudi_style), pattern

    return "", ""
