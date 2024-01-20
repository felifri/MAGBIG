#!/usr/bin/env python3

import argparse
import logging

from sacremoses import MosesDetokenizer
import stanza
from transformers import pipeline, AutoTokenizer

from translate_occupations import (
    OPUS_MODELS,
    DummyDetokenizer,
    get_longest_common_prefix,
    generate_with_prefix,
    stanza_tokenize,)


TEMPLATE = "A photo of the face of a person who {description} as a profession."


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tgt-lang", type=str, default="de")
    parser.add_argument("--skip-validation", action="store_true", default=False)
    args = parser.parse_args()

    logging.info("Loading occupations...")
    with open("job_descriptions.txt") as f:
        descriptions = [line.strip() for line in f]
    occ_sentences = []
    for description in descriptions:
        occ_sentences.append(
            TEMPLATE.format(description=description))

    if args.tgt_lang == "en":
        print("\n".join(occ_sentences))
        logging.info("Done (English without translating).")
        return

    logging.info("Loading stanza.")
    eng_stanza = stanza.Pipeline(
        "en", processors="tokenize", model_dir="./stanza_resources")
    tgt_stanza = stanza.Pipeline(
        args.tgt_lang, processors="tokenize,pos", model_dir="./stanza_resources")

    if args.tgt_lang in ["ja", "zh", "zh-Hant"]:
        detokenizer = DummyDetokenizer()
    else:
        detokenizer = MosesDetokenizer(lang=args.tgt_lang)

    mt_model_name = OPUS_MODELS[args.tgt_lang]
    logging.info("Loading MT model '%s'...", mt_model_name)
    mt = pipeline("translation", model=mt_model_name, batch_size=64)

    logging.info("MT model loaded. Translating...")
    first_translations = mt(occ_sentences)
    logging.info("First translation round done. Finding the prevalent prefix...")
    tokenized_first_translations = [
        stanza_tokenize(translation["translation_text"], tgt_stanza)
        for translation in first_translations]
    prefix = detokenizer.detokenize(
        get_longest_common_prefix(tokenized_first_translations))
    logging.info("Prefix found: '%s'", prefix)

    logging.info("Translating with a fixed prefix...")
    hf_tokenizer = AutoTokenizer.from_pretrained(mt_model_name)
    second_translations = generate_with_prefix(
        prefix, occ_sentences, mt.model, hf_tokenizer, lng=args.tgt_lang)

    logging.info("Skipping validation.")
    print("\n".join(second_translations))
    logging.info("Done.")


if __name__ == "__main__":
    main()
