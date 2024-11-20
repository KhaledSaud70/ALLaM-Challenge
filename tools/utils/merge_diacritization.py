from diacritization_evaluation.util import extract_haraqat, combine_txt_and_haraqat


def merge_diacritization(automatic_text, manual_text):
    _, manual_base_text, manual_diacritics = extract_haraqat(manual_text)
    _, automatic_base_text, automatic_diacritics = extract_haraqat(automatic_text)

    if len(manual_diacritics) != len(automatic_diacritics):
        raise ValueError(f"Diacritic lengths do not match: {len(manual_diacritics)} vs {len(automatic_diacritics)}")

    for position in range(len(manual_diacritics)):
        if manual_diacritics[position]:
            automatic_diacritics[position] = manual_diacritics[position]

    result = combine_txt_and_haraqat(automatic_base_text, automatic_diacritics)
    return result
