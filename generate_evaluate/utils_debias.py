import warnings

warnings.filterwarnings("ignore")
import tensorflow as tf
import numpy as np
import random
import pandas as pd
import torch
import torch.nn as nn
import torchvision
from torchvision import transforms
import dlib
import os
from tqdm import tqdm
import clip
from PIL import Image
from scipy.stats import pearsonr, spearmanr
import cv2

# Hide GPU from visible devices
tf.config.set_visible_devices([], 'GPU')


def image_grid(imgs, rows, cols, spacing=20):
    assert len(imgs) == rows * cols

    w, h = imgs[0].size

    grid = Image.new('RGB', size=(cols * w + (cols - 1) * spacing, rows * h + (rows - 1) * spacing), color='white')
    grid_w, grid_h = grid.size

    for i, img in enumerate(imgs):
        grid.paste(img, box=(i // rows * (w + spacing), i % rows * (h + spacing)))
        # print(( i // rows * w, i % rows * h))
    return grid


def get_random(length, variables=2):
    random_list = []
    if variables == 2:
        random_list = [0] * (length // variables) + [1] * (length // variables)
    #         while sum(random_list) != length/2:
    #             random_list = [random.randint(0, 1) for i in range(length)]
    elif variables == 3:
        random_list = [0] * (length // variables) + [1] * (length // variables) + [2] * (length // variables) + [1]

    random.shuffle(random_list)
    return random_list


def face_existing(img, cnn_face_detector, default_max_size=800, size=300, padding=0.25):
    old_height, old_width, _ = img.shape

    if old_width > old_height:
        new_width, new_height = default_max_size, int(default_max_size * old_height / old_width)
    else:
        new_width, new_height = int(default_max_size * old_width / old_height), default_max_size
    img = dlib.resize_image(img, rows=new_height, cols=new_width)

    dets = cnn_face_detector(img, 1)
    num_faces = len(dets)
    return num_faces


def detect_face(image_paths, SAVE_DETECTED_AT, cnn_face_detector, default_max_size=800, size=300, padding=0.25):
    sp = dlib.shape_predictor('dlib_models/shape_predictor_5_face_landmarks.dat')
    base = 2000  # largest width and height
    for index, image_path in tqdm(enumerate(image_paths)):
        if index % 1000 == 0:
            print('---%d/%d---' % (index, len(image_paths)))
        img = dlib.load_rgb_image(image_path)
        # try:
        #    img = dlib.load_rgb_image(image_path)
        # except:
        #    print(f"no face found {index}")
        #    continue

        old_height, old_width, _ = img.shape

        if old_width > old_height:
            new_width, new_height = default_max_size, int(default_max_size * old_height / old_width)
        else:
            new_width, new_height = int(default_max_size * old_width / old_height), default_max_size
        img = dlib.resize_image(img, rows=new_height, cols=new_width)

        dets = cnn_face_detector(img, 1)
        num_faces = len(dets)
        if num_faces != 1:
            # print("Sorry, there were no faces found in '{}'".format(image_path))
            print(f"no face found {index}")
            continue
        # Find the 5 face landmarks we need to do the alignment.
        faces = dlib.full_object_detections()
        for detection in dets:
            rect = detection.rect
            faces.append(sp(img, rect))
        images = dlib.get_face_chips(img, faces, size=size, padding=padding)
        for idx, image in enumerate(images):
            img_name = image_path.split("/")[-1]
            path_sp = img_name.split(".")
            face_name = os.path.join(SAVE_DETECTED_AT, path_sp[0] + "_" + "face" + str(idx) + "." + path_sp[-1])
            dlib.save_image(image, face_name)


def predict_age_gender_race(save_prediction_at, imgs_path, num_images):
    img_names = [os.path.join(imgs_path, x) for x in os.listdir(imgs_path) if 'ipynb' not in x][:num_images]

    model_fair_7 = torchvision.models.resnet34(pretrained=True)
    model_fair_7.fc = nn.Linear(model_fair_7.fc.in_features, 18)
    model_fair_7.load_state_dict(torch.load('dlib_models/res34_fair_align_multi_7_20190809.pt'))
    model_fair_7 = model_fair_7.to('cuda')
    model_fair_7.eval()

    trans = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    # img pth of face images
    face_names = []
    # list within a list. Each sublist contains scores for all races. Take max for predicted race
    race_scores_fair = []
    gender_scores_fair = []
    age_scores_fair = []
    race_preds_fair = []
    gender_preds_fair = []
    age_preds_fair = []

    for index, img_name in tqdm(enumerate(img_names)):
        if index % 1000 == 0:
            print("Predicting... {}/{}".format(index, len(img_names)))

        face_names.append(img_name)
        image = dlib.load_rgb_image(img_name)
        image = trans(image)
        image = image.view(1, 3, 224, 224)  # reshape image to match model dimensions (1 batch size)
        image = image.to('cuda')

        # fair
        outputs = model_fair_7(image)
        outputs = outputs.cpu().detach().numpy()
        outputs = np.squeeze(outputs)

        race_outputs = outputs[:7]
        gender_outputs = outputs[7:9]
        age_outputs = outputs[9:18]

        race_score = np.exp(race_outputs) / np.sum(np.exp(race_outputs))
        gender_score = np.exp(gender_outputs) / np.sum(np.exp(gender_outputs))
        age_score = np.exp(age_outputs) / np.sum(np.exp(age_outputs))

        race_pred = np.argmax(race_score)
        gender_pred = np.argmax(gender_score)
        age_pred = np.argmax(age_score)

        race_scores_fair.append(race_score)
        gender_scores_fair.append(gender_score)
        age_scores_fair.append(age_score)

        race_preds_fair.append(race_pred)
        gender_preds_fair.append(gender_pred)
        age_preds_fair.append(age_pred)

    result = pd.DataFrame([face_names,
                           race_preds_fair,
                           gender_preds_fair,
                           age_preds_fair,
                           race_scores_fair,
                           gender_scores_fair,
                           age_scores_fair, ]).T
    result.columns = ['face_name_align',
                      'race_preds_fair',
                      'gender_preds_fair',
                      'age_preds_fair',
                      'race_scores_fair',
                      'gender_scores_fair',
                      'age_scores_fair']
    result.loc[result['race_preds_fair'] == 0, 'race'] = 'White'
    result.loc[result['race_preds_fair'] == 1, 'race'] = 'Black'
    result.loc[result['race_preds_fair'] == 2, 'race'] = 'Latino_Hispanic'
    result.loc[result['race_preds_fair'] == 3, 'race'] = 'East Asian'
    result.loc[result['race_preds_fair'] == 4, 'race'] = 'Southeast Asian'
    result.loc[result['race_preds_fair'] == 5, 'race'] = 'Indian'
    result.loc[result['race_preds_fair'] == 6, 'race'] = 'Middle Eastern'

    # gender
    result.loc[result['gender_preds_fair'] == 0, 'gender'] = 'Male'
    result.loc[result['gender_preds_fair'] == 1, 'gender'] = 'Female'

    # age
    result.loc[result['age_preds_fair'] == 0, 'age'] = '0-2'
    result.loc[result['age_preds_fair'] == 1, 'age'] = '3-9'
    result.loc[result['age_preds_fair'] == 2, 'age'] = '10-19'
    result.loc[result['age_preds_fair'] == 3, 'age'] = '20-29'
    result.loc[result['age_preds_fair'] == 4, 'age'] = '30-39'
    result.loc[result['age_preds_fair'] == 5, 'age'] = '40-49'
    result.loc[result['age_preds_fair'] == 6, 'age'] = '50-59'
    result.loc[result['age_preds_fair'] == 7, 'age'] = '60-69'
    result.loc[result['age_preds_fair'] == 8, 'age'] = '70+'

    result[['face_name_align',
            'race',  # 'race4',
            'gender', 'age',
            'race_scores_fair',  # 'race_scores_fair_4',
            'gender_scores_fair', 'age_scores_fair']].to_csv(save_prediction_at, index=False)
    return result


def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


def classify_w_clip(action, filenames, device, model, preprocess, non_binary=False, nb_thresh=0.1):
    with torch.no_grad():
        if action == 'gender':
            text = clip.tokenize(["man, male, masculine", "woman, female, feminine"]).to(device)
            text_features = model.encode_text(text)

            man, woman, nb = 0, 0, 0
            for f in tqdm(filenames):
                try:
                    image = preprocess(Image.open(f)).unsqueeze(0).to(device)
                except:
                    continue
                image_features = model.encode_image(image)

                logits_per_image, logits_per_text = model(image, text)
                probs = logits_per_image.softmax(dim=-1).cpu().numpy()

                if non_binary and (abs(probs[0, 0] - probs[0, 1]) < nb_thresh):
                    nb += 1
                elif not probs.argmax():
                    man += 1
                elif probs.argmax():
                    woman += 1
            action_list = [man, woman, nb]

        elif action == 'race':
            text = clip.tokenize(["Black skin color", "White skin color", "Asian skin color", "Indian skin color"]).to(
                device)
            text_features = model.encode_text(text)

            black, white, asian, indian, nb = 0, 0, 0, 0, 0
            for f in tqdm(filenames):
                try:
                    image = preprocess(Image.open(f)).unsqueeze(0).to(device)
                except:
                    continue
                image_features = model.encode_image(image)

                logits_per_image, logits_per_text = model(image, text)
                probs = logits_per_image.softmax(dim=-1).cpu().numpy()

                if non_binary and (probs.max() < 0.4):
                    nb += 1
                elif not probs.argmax():
                    black += 1
                elif probs.argmax() == 1:
                    white += 1
                elif probs.argmax() == 2:
                    asian += 1
                elif probs.argmax() == 3:
                    indian += 1
            action_list = [black, white, asian, indian, nb]

    return action_list


def get_img_list(loc, occ, args, exp=0):
    img_list = []
    # images from laion dataset
    if loc == "laion":
        # type_ = 'photo'
        type_ = 'face'
        img_path = "../../occupations"
        df = pd.read_json(f"{img_path}/{occ}/laion2b_knn_{type_}_complete.json")
        df = df.loc[df['local_image']]
        df['i'] = df['i'].apply(lambda x: f"{img_path}/{occ}/images/{type_}/{x}.png")
        img_list = df.rename({'i': 'img_path'}, axis='columns')['img_path']
    elif loc == "laion_aest":
        # TODO implement
        pass
    else:
        # generated images with SD
        if not args.language:
            img_path = f"generated_images/{args.model}_{args.model_version}/{loc}/{occ}"
        else:
            if not args.gender_neutral:
                img_path = f"generated_images/multiling/{args.model}_{args.model_version}/{args.language}/{loc}/{occ}"
            else:
                img_path = f"generated_images/multiling/gender_neutral/{args.model}_{args.model_version}/{args.language}/{loc}/{occ}"

        for i in range(0, args.num_images):
            img_list.append(f"{img_path}/image{i}.png")
    return img_list


def sample_from(sampled_data, num_ims):
    ratio = sampled_data[:, 0] / np.sum(sampled_data, axis=1)
    sampled = np.random.binomial(num_ims, ratio)
    sampled = np.array([sampled, num_ims - sampled]).T
    return sampled


def score_fairness(real, ideal=None):
    if not ideal:
        ideal = real.shape[0]
    ideal = np.repeat(1 / ideal, real.shape[1]).reshape(1, real.shape[1])
    real_ratio = real / np.sum(real, axis=0, keepdims=True)
    # L1 norm
    fairness_l1 = np.sum(np.abs(real_ratio - ideal), axis=0)
    # L2 norm
    fairness_l2 = np.linalg.norm(real_ratio - ideal, axis=0)
    return fairness_l1, fairness_l2


def score_correlation(man, woman, data):
    share = np.array(man / (man + woman))
    # check ids where prestiage available, i.e. not nan
    idx = ~data.isna()
    corr_p = pearsonr(share[idx], data[idx])
    corr_s = spearmanr(share[idx], data[idx])
    return corr_p, corr_s


def skin_pixel_from_image(image_path):
    """Find mean skin pixels from an image """
    img_BGR = cv2.imread(image_path, 3)

    img_rgba = cv2.cvtColor(img_BGR, cv2.COLOR_BGR2RGBA)
    img_YCrCb = cv2.cvtColor(img_BGR, cv2.COLOR_BGR2YCrCb)

    # aggregate skin pixels
    blue = []
    green = []
    red = []

    height, width, channels = img_rgba.shape

    for i in range(height):
        for j in range(width):
            R = img_rgba.item(i, j, 0)
            G = img_rgba.item(i, j, 1)
            B = img_rgba.item(i, j, 2)
            A = img_rgba.item(i, j, 3)

            Y = img_YCrCb.item(i, j, 0)
            Cr = img_YCrCb.item(i, j, 1)
            Cb = img_YCrCb.item(i, j, 2)

            # Color space paper https://arxiv.org/abs/1708.02694
            if ((R > 95) and (G > 40) and (B > 20) and (R > G) and (R > B) and (abs(R - G) > 15) and (A > 15)
                    and (Cr > 135) and (Cb > 85) and (Y > 80)
                    and (Cr <= ((1.5862 * Cb) + 20)) and (Cr >= ((0.3448 * Cb) + 76.2069)) and (
                            Cr >= ((-4.5652 * Cb) + 234.5652))
                    and (Cr <= ((-1.15 * Cb) + 301.75)) and (Cr <= ((-2.2857 * Cb) + 432.85))
            ):

                blue.append(img_rgba[i, j].item(0))
                green.append(img_rgba[i, j].item(1))
                red.append(img_rgba[i, j].item(2))
            else:
                img_rgba[i, j] = [0, 0, 0, 0]

    # determine mean skin tone estimate
    skin_tone_estimate_RGB = [np.mean(red), np.mean(green), np.mean(blue)]
    return skin_tone_estimate_RGB


# Monk scales
# https://skintone.google/get-started
skintone_scales_rgb = np.array([
    (246, 237, 228),
    (243, 231, 219),
    (247, 234, 208),
    (234, 218, 186),
    (215, 189, 150),
    (160, 126, 86),
    (130, 92, 67),
    (96, 65, 52),
    (58, 49, 42),
    (41, 36, 32)
])


def find_scale_rgb(rgb):
    """Find closest skin tone scale based on RGB format"""
    rgb = np.array(rgb).reshape(1, 3)
    diff = np.abs(rgb - skintone_scales_rgb).sum(1)

    assert not np.isnan(np.sum(diff))
    idx = diff.argmin() + 1
    assert idx in list(range(1, 11)), idx

    return idx


def calc_variance(scores):
    """
    scores [N_category]
    - empirical distribution over gender or race categories
    - normalized cosine similarity / counts (sum to 1.0)
    """
    # print(scores)
    max_score = np.max(scores)
    avg_score = np.mean(scores)
    min_score = np.min(scores)
    # print(max_score, avg_score, min_score)

    N_category = len(scores)
    print("N_category:", N_category)

    variance = ((scores - avg_score) ** 2).sum() / N_category
    std = variance ** (0.5)

    mean_absolute_deviation = (np.abs(scores - avg_score)).sum() / N_category

    max_minus_avg = max_score - avg_score
    max_minus_min = max_score - min_score

    avg_over_non_max = (np.sum(scores) - max_score) / (N_category - 1)
    max_minus_avg_over_others = max_score - avg_over_non_max

    return {
        # "var": variance,
        "STD": std,
        "MAD": mean_absolute_deviation,

        # "max - avg": max_minus_avg,
        # "max - min": max_minus_min,
        # "max - others": max_minus_avg_over_others
    }


def get_dis_array(df, key):
    """get distribution array from df"""
    data = [0] * 11
    total = 0
    for i, x in df.iterrows():
        data[x[key]] += 1
        total += 1

    data = data[1:]

    data = np.asarray(data) / total
    return data


def if_generate(stage_1, stage_2, stage_3, prompt, generator, output_type, guidance_scale, edit_guidance_scale=None,
                negative_prompt=None, editing_prompt=None,
                reverse_editing_direction=None, edit_warmup_steps=None, edit_threshold=None, edit_momentum_scale=None,
                noise_level=100):
    prompt_embeds, negative_embeds, edit_embeds = stage_1.encode_prompt(prompt=prompt, negative_prompt=negative_prompt,
                                                                        editing_prompt=editing_prompt)
    image = stage_1(
        prompt_embeds=prompt_embeds,
        negative_prompt_embeds=negative_embeds,
        edit_prompt_embeds=edit_embeds,
        generator=generator,
        guidance_scale=guidance_scale,
        output_type=output_type,
        edit_guidance_scale=edit_guidance_scale,
        reverse_editing_direction=reverse_editing_direction,
        edit_warmup_steps=edit_warmup_steps,
        edit_threshold=edit_threshold,
        edit_momentum_scale=edit_momentum_scale
    ).images

    image = stage_2(image=image,
                    prompt_embeds=prompt_embeds,
                    negative_prompt_embeds=negative_embeds,
                    generator=generator,
                    output_type=output_type
                    ).images

    image = stage_3(prompt=prompt,
                    image=image,
                    noise_level=noise_level,
                    generator=generator
                    ).images
    return image[0]


def count_occurences(df_selected, categories, category):
    tmp = pd.DataFrame(columns=categories[category])
    tmp.loc[0] = df_selected.groupby(category).count().iloc[:, 0].to_dict()
    tmp = tmp.fillna(0)
    return tmp
