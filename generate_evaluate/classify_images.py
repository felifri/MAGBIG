import pandas as pd
from utils_debias import detect_face, predict_age_gender_race, ensure_dir, classify_w_clip, get_img_list
import dlib
import argparse
import clip
import torch

parser = argparse.ArgumentParser(description='Bias in Diffusion Evaluation')
parser.add_argument('--model', default='SD', type=str,
                    help='which model to evaluate')
parser.add_argument('--model_version', default='1-5', type=str,
                    help='which version of this model to evaluate')
parser.add_argument('--mode', default='generated', type=str,
                    choices=['laion', 'sega', 'generated', 'baseline', 'baseline_ext', 'baseline_neg'],
                    help='which images to evaluate')
parser.add_argument('--classifier', default='fairface', type=str, choices=['fairface', 'clip'],
                    help='which classifier to use for evaluation')
parser.add_argument('--dataset', default='occupations', type=str, choices=['occupations', 'adjectives'],
                    help='which dataset to evaluate')
parser.add_argument('--action', default='gender', type=str, choices=['gender', 'skin'],
                    help='what category to evaluate')
parser.add_argument('--non-binary', default=False, type=bool,
                    help='only binary evaluation?')
parser.add_argument('--num_images', default=100, type=int,
                    help='how many images to generate')
parser.add_argument('--language', default='', type=str,
                    choices=['', 'english', 'arabic', 'chinese_simplified', 'chinese_traditional', 'spanish', 'italian',
                             'german', 'german_star', 'korean', 'russian', 'french', 'japanese'],
                    help='what category to evaluate')
parser.add_argument('--gender_neutral', default='', type=str,
                    help='whether to evaluate gender-neutral prompts')
args = parser.parse_args()

if args.dataset == 'occupations':
    data = pd.read_csv('prompts/gender_neutral/occ_english.csv')['name']
elif args.dataset == 'adjectives':
    with open(f'prompts/adjectives.txt') as f:
        data = [line.split("\n", 1)[0] for line in f]

if args.classifier == 'fairface':
    dlib.DLIB_USE_CUDA = True
    cnn_face_detector = dlib.cnn_face_detection_model_v1('dlib_models/mmod_human_face_detector.dat')
elif args.classifier == 'clip':
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-L/14", device=device)
    model.eval()

if args.mode == 'laion':
    txt_file = open(f'results/{args.dataset}_{args.classifier}_{args.mode}.txt', 'w+')
    for d in data:
        man, woman = 0, 0
        # img_list = get_img_list(args.mode, d+' person', args)
        img_list = get_img_list(args.mode, d, args)
        if args.classifier == 'fairface':
            pth_ff = f'detected_faces/{args.mode}'
            SAVE_DETECTED_AT = f'{pth_ff}/{d}'
            ensure_dir(SAVE_DETECTED_AT)
            detect_face(img_list, SAVE_DETECTED_AT, cnn_face_detector)
            result = predict_age_gender_race(f'{pth_ff}/{args.dataset}_fairface_{d}.csv', SAVE_DETECTED_AT,
                                             args.num_images)
            man = len(result[result[args.action] == 'Male'])
            woman = len(result[result[args.action] == 'Female'])
        elif args.classifier == 'clip':
            man, woman, _ = classify_w_clip(args.action, img_list, device, model, preprocess, args.non_binary)

        txt_file.write(f'{d}\n')
        txt_file.write(f'man: {man}, woman: {woman}\n')
    txt_file.close()

else:
    if not args.language:
        pth = f"results/{args.model}_{args.model_version}"
        pth_ff = f'detected_faces/{args.model}_{args.model_version}/{args.mode}'
    else:
        if not args.gender_neutral:
            pth = f"results/multiling/{args.model}_{args.model_version}/{args.language}"
            pth_ff = f'detected_faces/multiling/{args.model}_{args.model_version}/{args.language}/{args.mode}'
        else:
            pth = f"results/multiling/gender_neutral/{args.model}_{args.model_version}/{args.language}"
            pth_ff = f'detected_faces/multiling/gender_neutral/{args.model}_{args.model_version}/{args.language}/{args.mode}'

    ensure_dir(pth)
    txt_file = open(f'{pth}/{args.dataset}_{args.classifier}_{args.mode}.txt', 'w+')
    for d in data:
        man, woman = 0, 0
        img_list = get_img_list(args.mode, d, args)
        if args.classifier == 'fairface':
            SAVE_DETECTED_AT = f'{pth_ff}/{d}'
            ensure_dir(SAVE_DETECTED_AT)
            detect_face(img_list, SAVE_DETECTED_AT, cnn_face_detector)
            result = predict_age_gender_race(f'{pth_ff}/results_fairface_{d}.csv', SAVE_DETECTED_AT, args.num_images)
            man = len(result[result[args.action] == 'Male'])
            woman = len(result[result[args.action] == 'Female'])
        elif args.classifier == 'clip':
            man, woman, _ = classify_w_clip(args.action, img_list, device, model, preprocess, args.non_binary)
        txt_file.write(f'{d}\n')
        txt_file.write(f'man: {man}, woman: {woman}\n')
    txt_file.close()
