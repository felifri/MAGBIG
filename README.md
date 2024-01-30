# MAGBIG

This is the repository for MAGBIG (**M**ultilingual **A**ssessment of **G**ender **B**ias in **I**mage **G**eneration) proposed in "Multilingual Text-to-Image Generation Magnifies Gender Stereotypes and Prompt Engineering May Not Help You".

You can find a Python package requirements file at `translate_script\requirements.txt` and our set of prompts for all prompt types and languages in the folder `prompt`. 

### Generating and Evaluating Images
You can generate images for these prompts using `python generate_evaluate\generate_images.py`. After that, you can classify them with `python generate_evaluate\classify_images.py`. For the evaluation, you can use `python generate_evaluate\exp_max_unfairness.py` for bias and `python generate_evaluate\CLIPscore.py` for text-to-image alignment with CLIP. These Python scripts also reproduce our results.


### Generating and Translating prompts
To reproduce our prompts, you can run your bash script `translate_script\run.sh` or modify it to compute your own translations for new prompts and languages.

Please cite our work if you find it helpful.


### Easy use of benchmark
You can also use our benchmark in the huggingface dataset library: https://huggingface.co/datasets/felfri/MAGBIG


### citation
If you like or use our work, please consider citing us.
```
@misc{friedrich2024multilingual,
      title={Multilingual Text-to-Image Generation Magnifies Gender Stereotypes and Prompt Engineering May Not Help You}, 
      author={Felix Friedrich and Katharina Hämmerl and Patrick Schramowski and Jindrich Libovicky and Kristian Kersting and Alexander Fraser},
      year={2024},
      eprint={2401.16092},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```
