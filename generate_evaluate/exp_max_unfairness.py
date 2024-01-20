import pandas as pd
import numpy as np
from utils_debias import score_fairness, sample_from
import argparse
import re

parser = argparse.ArgumentParser(description='Bias in Diffusion Evaluation')
parser.add_argument('--model', default='SD', type=str,
                    help='which model to evaluate')
parser.add_argument('--model_version', default='1-5', type=str,
                    help='which version of this model to evaluate')
parser.add_argument('--classifier', default='fairface', type=str, choices=['fairface', 'clip'],
                    help='which classifier to use for evaluation')
parser.add_argument('--dataset', default='occupations', type=str, choices=['occupations', 'adjectives'],
                    help='which dataset to evaluate')
parser.add_argument('--num_ims', default=10, type=int,
                    help='how many images to evaluate for exp max')
parser.add_argument('--num_occs', default=10, type=int,
                    help='how many occupations to evaluate for exp max')
parser.add_argument('--num_samples', default=1000, type=int,
                    help='how many samples to evaluate for exp max')
args = parser.parse_args()

categories = {
    'race': ['White', 'Black', 'Latino_Hispanic', 'East Asian', 'Southeast Asian', 'Indian', 'Middle Eastern'],
    'gender': ['Male', 'Female'],
    'age': ['0-2', '3-9', '10-19', '20-29', '30-39', '40-49', '50-59', '60-69', '70+'],
    'skin_tone': list(range(10))}

# locs = ['laion', 'generated', 'sega', 'baseline', 'baseline_ext', 'baseline_neg']
locs = ['generated', 'sega', 'baseline', 'baseline_ext', 'baseline_neg']
if args.dataset == 'occupations':
    with open('prompts/final_occupations.txt') as f:
        data = [line.split("   ", 1)[0] for line in f][1:]
elif args.dataset == 'adjectives':
    with open(f'prompts/adjectives.txt') as f:
        data = [line.split("\n", 1)[0] for line in f]

len_ = len(data)

# use same stratification for all methods?
sample_list = [np.random.choice(range(len_), size=args.num_occs, replace=False) for _ in range(args.num_samples)]

######### gender ##############

df_f = []
for loc in locs:
    fairness_scores_l1 = []
    fairness_scores_l2 = []
    results = pd.read_csv(f'results/{args.model}_{args.model_version}/{args.dataset}_{args.classifier}_{loc}.txt',
                          sep=' ', skiprows=range(0, len_ * 2, 2), header=None)
    results.iloc[:, 1] = results.iloc[:, 1].apply(lambda x: int(re.findall(r'\d+', str(x))[0]))
    results.iloc[:, 3] = results.iloc[:, 3].apply(lambda x: int(re.findall(r'\d+', str(x))[0]))
    # overall mean deviation
    l1, l2 = score_fairness(np.array([results.iloc[:, 1], results.iloc[:, 3]]))
    mean_l1, std_l1 = np.mean(l1), np.std(l1)
    mean_l2, std_l2 = np.mean(l2), np.std(l2)
    # sampled deviation, user expectation
    for ids in sample_list:
        sampled_data = results.iloc[ids, [1, 3]].to_numpy()
        sampled_data = sample_from(sampled_data, args.num_ims)
        l1, l2 = score_fairness(np.array([sampled_data[:, 0], sampled_data[:, 1]]))
        fairness_scores_l1.append(np.max(l1))
        fairness_scores_l2.append(np.max(l2))
    exp_max_l1, exp_std_l1 = np.mean(fairness_scores_l1), np.std(fairness_scores_l1)
    exp_max_l2, exp_std_l2 = np.mean(fairness_scores_l2), np.std(fairness_scores_l2)
    df_f.append([loc, mean_l1, std_l1, exp_max_l1, exp_std_l1, mean_l2, std_l2, exp_max_l2, exp_std_l2])

# add random result as reference
rndm = np.random.normal(0.5, 0.1, len(results))
results = np.array([rndm, 1 - rndm]).T
l1, l2 = score_fairness(np.array([results[:, 0], results[:, 1]]))
mean_l1, std_l1 = np.mean(l1), np.std(l1)
mean_l2, std_l2 = np.mean(l2), np.std(l2)

# sampled deviation, user expectation
for ids in sample_list:
    sampled_data = sample_from(results, args.num_ims)
    l1, l2 = score_fairness(np.array([sampled_data[:, 0], sampled_data[:, 1]]))
    fairness_scores_l1.append(np.max(l1))
    fairness_scores_l2.append(np.max(l2))
exp_max_l1, exp_std_l1 = np.mean(fairness_scores_l1), np.std(fairness_scores_l1)
exp_max_l2, exp_std_l2 = np.mean(fairness_scores_l2), np.std(fairness_scores_l2)
df_f.append(['random', mean_l1, std_l1, exp_max_l1, exp_std_l1, mean_l2, std_l2, exp_max_l2, exp_std_l2])
df_f = pd.DataFrame(df_f, columns=['location', 'mean_l1', 'std_l1', 'exp_max_l1', 'exp_std_l1', 'mean_l2', 'std_l2',
                                   'exp_max_l2', 'exp_std_l2'])
df_f = df_f.round(decimals=2)
df_f.to_csv(f'results/{args.model}_{args.model_version}/fairness_{args.classifier}_{args.dataset}.csv', index=False)
