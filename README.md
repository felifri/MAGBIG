# MAGBIG

This is the repository for MAGBIG (**M**ultilingual **A**ssessment of **G**ender **B**ias in **I**mage **G**eneration) proposed in "Multilingual Text-to-Image Generation Magnifies Gender Stereotypes and Prompt Engineering May Not Help You"

You can find our set of prompts for all prompt types and languages in the folder `prompt`.


### Generating and Evaluating Images
You can generate images for these prompts using `python generate_images.py`. After that, you can classify them with `python classify_images.py`. For the evaluation, you can use `python exp_max_unfairness.py` for bias and `python CLIPscore.py` for text-to-image alignment with CLIP. These python scripts also reproduce our results.


### Generating and Translating prompts
To reproduce our prompts, you can run your bash script `run.sh` or modify it to compute your own translations for new prompts and languages.

Please cite our work if you find it helpful.
