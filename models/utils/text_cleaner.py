import re
from itertools import product

# Compile whitespace regex for performance
_whitespace_re = re.compile(r"\s+")


# Function to collapse multiple whitespaces into a single space
def collapse_whitespace(text):
    return re.sub(_whitespace_re, " ", text)


# Arabic diacritics and characters
HARAQAT = ["ْ", "ّ", "ٌ", "ٍ", "ِ", "ً", "َ", "ُ"]
ARAB_CHARS = "ىعظحرسيشضق ثلصطكآماإهزءأفؤغجئدةخوبذتن"
VALID_ARABIC = HARAQAT + list(ARAB_CHARS)

# Define different types of Arabic diacritics
harakat = ["\u0650", "\u064e", "\u064f"]  # kasra, fatha, damma
sukun = ["\u0652"]  # sukun
mostly_saken = ["\u0627", "\u0648", "\u0649", "\u064a"]  # alef, waw, alef maqsurah, ya'a
always_saken = ["\u0627", "\u0649"]  # alef, alef maqsurah

# Tanween, shadda, and all tashkeel characters
tnween_chars = ["\u064c", "\u064d", "\u064b"]  # damm, kasra, fatha tanween
shadda_chars = ["\u0651"]
all_tashkeel = harakat + tnween_chars + sukun + shadda_chars

# All Arabic characters and permissible characters
all_chars = list("إةابتثجحخدذرزسشصضطظعغفقكلمنهويىأءئؤ ")
prem_chars = harakat + sukun + mostly_saken + tnween_chars + shadda_chars + all_chars


# Function to check invalid tashkeel combinations
def not_valid_tashkeel_comb(comb):
    all_combinations = list(product(harakat + sukun + tnween_chars, repeat=2)) + list(
        product(shadda_chars + sukun, repeat=2)
    )
    return comb in all_combinations or comb[::-1] in all_combinations


# Function to remove tanween on alef
def remove_tanween_on_alef(text):
    cleaned_text = ""
    for i in range(0, len(text)):
        # If there is shaddah or character followed by alef and tanween
        if (
            i < len(text) - 2
            and text[i] in all_chars + shadda_chars
            and text[i + 1] in always_saken
            and text[i + 2] == tnween_chars[2]
        ):
            cleaned_text += text[i] + tnween_chars[2]
        # Ignore current harakah if there is alef followed by tanween
        elif (
            i < len(text) - 2 and text[i] in harakat and text[i + 1] in always_saken and text[i + 2] == tnween_chars[2]
        ):
            cleaned_text += tnween_chars[2]
        # Drop tanween if it follows alef
        elif i > 0 and text[i] == tnween_chars[2] and text[i - 1] in always_saken:
            continue
        else:
            cleaned_text += text[i]
    return cleaned_text


# Function to remove starting harakah in text
def dont_start_by_harakah(text):
    for i, char in enumerate(text):
        if char not in all_tashkeel:
            return text[i:]
    return text


# Function to clean Arabic text and validate diacritics
def valid_arabic_cleaners(text):
    prev_text = text
    for _ in range(5):
        text = prev_text
        cleaned_text = ""
        # Filter out invalid Arabic characters
        text = "".join(filter(lambda char: char in VALID_ARABIC, text))
        text = collapse_whitespace(text)
        text = dont_start_by_harakah(text).strip()

        i, cnt = 0, 0
        len_text = len(text)

        while i < len_text:
            if text[i] in all_tashkeel:
                cnt += 1
            else:
                cnt = 0

            # Skip if more than 2 consecutive tashkeel
            if cnt > 2:
                i += 1
                continue

            # Remove consecutive tanween or sukun
            if i > 1 and text[i] in tnween_chars + sukun and text[i - 2] in tnween_chars + sukun:
                i += 1
                continue

            # Skip harakah followed by shaddah or tanween
            if i < len(text) - 1 and text[i] in harakat and text[i + 1] in tnween_chars + sukun + shadda_chars:
                i += 1
                continue

            # Skip harakah on space
            if i > 0 and text[i] in all_tashkeel and text[i - 1] == " ":
                i += 1
                continue

            # Skip invalid tashkeel combinations
            if not_valid_tashkeel_comb((text[i], text[i - 1])):
                i += 1
                continue

            # Skip harakah on alef or alef maqsura without preceding tashkeel
            if i > 1 and text[i] in harakat and text[i - 1] in always_saken:
                if text[i - 2] not in all_tashkeel:
                    cleaned_text += text[: i - 1] + text[i] + always_saken[always_saken.index(text[i - 1])]
                i += 1
                continue

            cleaned_text += text[i]
            i += 1

        # Remove tanween on alef
        cleaned_text = remove_tanween_on_alef(cleaned_text)
        cleaned_text = re.sub(r" +", " ", cleaned_text).strip()

        if prev_text == cleaned_text:
            break
        else:
            prev_text = cleaned_text

    return cleaned_text
