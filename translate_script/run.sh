#!/bin/bash

for LNG in ar de es fr it ru; do
    python3 translate_occupations.py --tgt-lang $LNG > occupations.$LNG
done

for LNG in ja zh; do
    python3 translate_occupations.py --tgt-lang $LNG --skip-validation > occupations.$LNG
done

for LNG in ar de es fr it ru ko ja zh; do
    python3 translate_descriptions.py --tgt-lang $LNG > descriptions.$LNG
done
