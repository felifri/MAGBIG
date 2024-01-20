#!/usr/bin/env python3

import argparse
from collections import Counter
import logging

from sacremoses import MosesDetokenizer
from simalign import SentenceAligner
import stanza
import torch
import torch.nn.functional as F
from transformers import pipeline, AutoTokenizer

# Logging with timestamps
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO)


TEMPLATE = "A photo of the face of {article} {occupation}."


OPUS_MODELS = {
     "ar": "Helsinki-NLP/opus-mt-tc-big-en-ar",
     "de": "gsarti/opus-mt-tc-big-en-de",
     "es": "Helsinki-NLP/opus-mt-tc-big-en-es",
     "fr": "Helsinki-NLP/opus-mt-tc-big-en-fr",
     "it": "Helsinki-NLP/opus-mt-tc-big-en-it",
     "ko": "Helsinki-NLP/opus-mt-tc-big-en-ko",

     # Only small models available
     "ru": "Helsinki-NLP/opus-mt-en-ru",
     "ja": "staka/fugumt-en-ja",
     "zh": "Helsinki-NLP/opus-mt-en-zh",
}

FINAL_PUNCTUATION = {
    "zh": "。",
    "ja": "。",
        }


def get_longest_common_prefix(sentences, majority_threshold=0.3):
    prefix = []
    total_count = len(sentences)
    max_sentence_len = max(map(len, sentences))
    for i in range(max_sentence_len):
        counter = Counter(
            sentence[i] for sentence in sentences if i < len(sentence))
        # Get the frequent token at this position
        token, count = counter.most_common(1)[0]
        if count < total_count * majority_threshold:
            # No majority
            break
        prefix.append(token)
        sentences = [sent for sent in sentences if sent[i] == token]

    return prefix


def fix_final_punctuation(sentences, lng):
    for sent in sentences:
        if sent[-1] != FINAL_PUNCTUATION.get(lng, "."):
            sent += FINAL_PUNCTUATION.get(lng, ".")


@torch.no_grad()
def generate_with_prefix(
        prefix, occupation_sent, model, tokenizer, lng, batch_size=32):

    translations = []
    for i in range(0, len(occupation_sent), batch_size):
        batch = occupation_sent[i:i + batch_size]

        mt_input = tokenizer(
            batch,
            return_tensors="pt", padding=True).to(model.device)

        mt_output = model.generate(
            **mt_input,
            prefix_allowed_tokens_fn=PrefixControlFn(prefix, tokenizer))

        translations.extend(
            tokenizer.batch_decode(mt_output, skip_special_tokens=True))
    fix_final_punctuation(translations, lng)
    return translations


@torch.no_grad()
def sample_translation_and_score(
        model, tokenizer, source_sent, lng, prefix=None, num_samples=100):
    mt_input = tokenizer(
        [source_sent], return_tensors="pt", padding=True)

    prefix_allowed_tokens_fn = None
    if prefix is not None:
        prefix_allowed_tokens_fn = PrefixControlFn(prefix, tokenizer)

    logging.info("Translating '%s'.", source_sent)
    mt_output = model.generate(
        **mt_input.to("cuda:1"),
        output_scores=True,
        return_dict_in_generate=True,
        do_sample=True, top_p=0.99,
        temperature=2.0,
        num_return_sequences=num_samples,
        prefix_allowed_tokens_fn=prefix_allowed_tokens_fn)
    all_scores = F.log_softmax(
        torch.stack(mt_output.scores, dim=1), dim=-1)

    logging.info("Computing scores.")
    decoded_scores = []
    for sent_tensor, scores in zip(mt_output.sequences, all_scores):
        tok_probs = scores.index_select(1, sent_tensor[1:]).diag()
        sent_len = (sent_tensor[1:] != tokenizer.pad_token_id).sum().item()
        sent_score = tok_probs[:sent_len].sum().item() / sent_len
        decoded_scores.append(sent_score)

    logging.info("Decoding strings.")
    decoded = tokenizer.batch_decode(
        mt_output.sequences, skip_special_tokens=True)
    fix_final_punctuation(decoded, lng)
    return decoded, decoded_scores


class PrefixControlFn:
    def __init__(self, prefix, hf_tokenizer):
        with hf_tokenizer.as_target_tokenizer():
            self.prefix = hf_tokenizer.encode(prefix)[:-1]
        self.tokenizer = hf_tokenizer
        self.vocab_size = hf_tokenizer.vocab_size

    def __call__(self, batch_id, input_ids):
        next_idx = input_ids.shape[-1] - 1
        if next_idx < len(self.prefix):
            return [self.prefix[next_idx]]
        return list(range(self.vocab_size))


class DummyDetokenizer:
    def detokenize(self, tokens):
        return "".join(tokens)


def stanza_tokenize(text, pipeline):
    if not text:
        return []
    return [
        token.text for sent in pipeline(text).sentences
        for token in sent.tokens]


def check_translation(
        eng_sent, tgt_sent, aligner, eng_stanza, tgt_stanza,
        face_token_id=4, photo_token_id=1,
        min_len_ratio=0.5, max_len_ratio=2.0,
        check_feminine_free=True):
    len_ratio = len(eng_sent) / len(tgt_sent)
    if len_ratio < min_len_ratio or len_ratio > max_len_ratio:
        return False

    tgt_doc = tgt_stanza(tgt_sent)
    if len(tgt_doc.sentences) != 1:
        return False
    eng_analyzed = [
        tok for sent in eng_stanza(eng_sent).sentences
        for tok in sent.tokens]
    tgt_analyzed = [tok for sent in tgt_doc.sentences for tok in sent.tokens]
    aligned = aligner.get_word_aligns(
        " ".join([token.text for token in eng_analyzed]),
        " ".join([token.text for token in tgt_analyzed]))["itermax"]

    if all(src_id != face_token_id for src_id, tgt_id in aligned):
        return False
    if all(src_id != photo_token_id for src_id, tgt_id in aligned):
        return False

    if not check_feminine_free:
        return True

    feminine_noun_exists = False
    tgt_tok_of_interest = [
        tgt for src, tgt in aligned if src >= len(eng_analyzed) - 2]
    for tgt_id in tgt_tok_of_interest:
        tok_dict = tgt_analyzed[tgt_id].to_dict()[0]
        if ("upos" in tok_dict and
            tok_dict["upos"] == "NOUN" and
            'feats' in tok_dict and
            'Gender=Fem' in tok_dict['feats']):
            feminine_noun_exists = True
    return not feminine_noun_exists


def materialize_template(occupations, template=TEMPLATE):
    occ_sentences = []
    for occupation in occupations:
        if occupation[0] in "aeio":
            article = "an"
        else:
            article = "a"
        occ_sentences.append(
            template.format(article=article, occupation=occupation))
    return occ_sentences


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tgt-lang", type=str, default="de")
    parser.add_argument("--skip-validation", action="store_true", default=False)
    args = parser.parse_args()

    logging.info("Loading occupations...")
    with open("occupations_masc.txt") as f:
        occupations = [line.strip() for line in f]
    occ_sentences = materialize_template(occupations)
    logging.info("Occupations loaded (%d).", len(occupations))

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
    mt = pipeline("translation", model=mt_model_name, batch_size=64, device=1)

    logging.info("MT model loaded. Loading sentence aligner...")
    aligner = SentenceAligner(
        model="xlmr", token_type="word", matching_methods="mai")

    logging.info("Sentence aligner loaded. Translating...")
    first_translations = mt(occ_sentences)
    logging.info("First translation round done. Filter translation for completenes.")

    filtered_first_translations = [
        trans["translation_text"]
        for eng_sent, trans in zip(occ_sentences, first_translations)
        if check_translation(
            eng_sent, trans["translation_text"],
            aligner, eng_stanza, tgt_stanza,
            max_len_ratio=5.0 if args.tgt_lang in ["zh", "ja"] else 2.0)]
    logging.info("Filtered first translations (%d/%d), find common prefix.",
        len(filtered_first_translations), len(occ_sentences))
    tokenized_first_translations = [
        stanza_tokenize(translation, tgt_stanza)
        for translation in filtered_first_translations]
    prefix = detokenizer.detokenize(
        get_longest_common_prefix(tokenized_first_translations))
    logging.info("Prefix found: '%s'", prefix)

    logging.info("Translating with a fixed prefix...")
    hf_tokenizer = AutoTokenizer.from_pretrained(mt_model_name)
    second_translations = generate_with_prefix(
        prefix, occ_sentences, mt.model, hf_tokenizer, args.tgt_lang)

    if args.skip_validation:
        logging.info("Skipping validation.")
        print("\n".join(second_translations))
        logging.info("Done.")
        return

    logging.info("Check if everything is in masculine gender.")
    final_translations = []
    for eng_sent, tgt_sent in zip(occ_sentences, second_translations):
        if not check_translation(
                eng_sent, tgt_sent, aligner, eng_stanza, tgt_stanza,
                max_len_ratio=5.0 if args.tgt_lang in ["zh", "ja"] else 2.0):
            logging.info("Retranslating with sampling: %s", eng_sent)
            decoded, decoded_scores = sample_translation_and_score(
                mt.model, hf_tokenizer, eng_sent, args.tgt_lang,
                num_samples=50, prefix=prefix)

            candidate_sentences = []
            for sent, score in zip(decoded, decoded_scores):
                if check_translation(
                    eng_sent, sent, aligner, eng_stanza, tgt_stanza,
                    max_len_ratio=5.0 if args.tgt_lang in ["zh", "ja"] else 2.0,
                    check_feminine_free=args.tgt_lang not in ["zh", "ja", "ko"]):
                    candidate_sentences.append((score, sent))

            if candidate_sentences:
                _, best_sent = max(candidate_sentences)
                tgt_sent = best_sent
            else:
                logging.info("No candidate found, use the original translation.")
        final_translations.append(tgt_sent)
    print("\n".join(final_translations))
    logging.info("Done.")


if __name__ == "__main__":
    main()
