import argparse
from econml.policy import DRPolicyTree
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import random
from sklearn.model_selection import train_test_split
from tqdm import tqdm

def get_ate(df, treatment_var='treat', outcome_var='voted14'):
    treat_mask = df[treatment_var] == 1
    control_mask = df[treatment_var] == 0

    mean_treat = df[outcome_var][treat_mask].mean()
    mean_control = df[outcome_var][control_mask].mean()

    ate = mean_treat - mean_control

    n_treat = treat_mask.sum()
    n_control = control_mask.sum()
    var_treat = df[outcome_var][treat_mask].var(ddof=1)
    var_control = df[outcome_var][control_mask].var(ddof=1)
    se = np.sqrt(var_treat / n_treat + var_control / n_control)

    results = {
        'mean_treatment': mean_treat,
        'mean_control': mean_control,
        'ate': ate,
        'se': se,
        'n_treat': n_treat,
        'n_control': n_control
    }
    return results

def compute_cate(df, group_vars, outcome_var='voted14', treat_var='treat'):
    cate = (
            df.groupby(group_vars, as_index=False)
            .apply(lambda x: pd.Series(get_ate(x, treatment_var=treat_var, outcome_var=outcome_var)), include_groups=False)
        )
    return cate


def policy_value_cate(train_cates, test_cates, cost):
    net_benefit = test_cates['ate'] - cost  
    treatment_assignment = train_cates['ate'] - cost > 0
    subgroup_sizes = test_cates['n_treat'] + test_cates['n_control']
    total_size = subgroup_sizes.sum()
    weighted_net_benefit = (net_benefit * treatment_assignment * subgroup_sizes).sum()
    return weighted_net_benefit / total_size

def get_policy_tree(df, policy_features, cost, max_depth=2):
    X = df[policy_features]

    T = df['treat'].astype('category')
    Y = df['voted14'] - cost

    max_depth = 2
    policy_tree = DRPolicyTree(max_depth=max_depth)
    policy_tree.fit(Y, T, X=X)
    return policy_tree

def get_leaf_assignments(policy_tree, X):
    recommended_treatments = policy_tree.predict(X)
    leaf_ids = policy_tree.policy_model_.tree_.apply(X.values.astype(np.float64))
    
    results_df = pd.DataFrame({
        'index': X.index,
        'leaf_id': leaf_ids,
    })

    leaf_treatment_dict = dict(zip(leaf_ids, recommended_treatments))
    results_df.set_index('index', inplace=True)
    
    return results_df, leaf_treatment_dict

def get_leaf_cates(policy_tree, df, cate_features):
    leaf_assignments, leaf_treatment_dict = get_leaf_assignments(policy_tree, df[cate_features])
    new_df = df.copy()
    new_df = new_df.merge(
        leaf_assignments[['leaf_id']], 
        left_index=True, right_index=True, how='left'
    )
    leaf_cates = compute_cate(new_df, ['leaf_id'])
    leaf_cates['recommended_treatment'] = 0
    for k, v in leaf_treatment_dict.items():
        leaf_cates.loc[leaf_cates['leaf_id'] == k, 'recommended_treatment'] = v
    return leaf_cates

def policy_value_tree(leaf_cates, cost):
    net_benefit = leaf_cates['ate'] - cost
    treat_mask = leaf_cates['recommended_treatment'] == 1
    subgroup_sizes = leaf_cates['n_treat'] + leaf_cates['n_control']
    total_size = leaf_cates['n_treat'].sum() + leaf_cates['n_control'].sum()
    weighted_net_benefit = (net_benefit * treat_mask * subgroup_sizes).sum()
    return weighted_net_benefit / total_size

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', type=int, required=False, dest='file_number')
    args = parser.parse_args()
    file_number = args.file_number

    df = pd.read_stata(f'gerber/gerber_generalizability_2/PublicReplicationData.dta')
    df = df[df['state'] == 'TX']

    # cate_features = ['d_race_b', 'd_race_h', 'd_race_o', 'd_race_w', 'd_female', 'd_notfem']
    cate_features = ['voted06', 'voted08', 'voted10', 'voted12']
    # policy_features = cate_features + ['voted06', 'voted08', 'voted09', 'voted10', 'voted11', 'voted12', 'voted13', 'i_age',
    #    'age_miss', 'age2', 'flag_hhid_mult_hhid', 'flag_hhid_mult_z',
    #    'flag_drop_hhid', 'vote_hist', 'state_median', 'vh_stratum', 'vhblw',
    #    'vhavg', 'vhabv', 'd_married', 'd_unmarried', 'd_hhsize1',
    #    'd_hhsize2', 'd_hhsize3', 'd_hhsize4']

    all_indices = np.arange(df.shape[0])
    B = 500
    cc = np.linspace(0,0.2,25)
    target_c = 0.1
    results = {}

    pv_cates_means = []
    pv_trees_means = []
    pv_cates_sds = []
    pv_trees_sds = []
    ates = []

    states = df['state'].unique()

    results_rows = []
    for i,state in enumerate(states):
        df_state = df[df['state'] == state]
        pv_cates = [[] for _ in range(len(cc))]
        pv_trees = [[] for _ in range(len(cc))]
        ates = [[] for _ in range(len(cc))]
        for _ in tqdm(range(B)):
            train, test = train_test_split(df_state, train_size=0.7, shuffle=True)
            # train = df_state
            # test = df_state
            train_ate = get_ate(train)['ate']
            train_cates = compute_cate(train, cate_features)
            test_cates = compute_cate(test, cate_features)
            policy_tree = get_policy_tree(train, cate_features, target_c, max_depth=3)
            leaf_cates = get_leaf_cates(policy_tree, test, cate_features)

            for i,c in enumerate(cc):
                pv_cate = policy_value_cate(train_cates, test_cates, cost=c)
                pv_tree = policy_value_tree(leaf_cates, cost=c)
                pv_ate = get_ate(test)['ate'] - c if train_ate > c else 0

                pv_cates[i].append(pv_cate)
                pv_trees[i].append(pv_tree)

                train_net_benefit = train_cates['ate'] - c  
                train_subgroup_sizes = train_cates['n_treat'] + train_cates['n_control']
                train_total_size = train_subgroup_sizes.sum()
                train_weighted_net_benefit = (train_net_benefit * train_subgroup_sizes).sum() / train_total_size 
                net_benefit = test_cates['ate'] - c  
                subgroup_sizes = test_cates['n_treat'] + test_cates['n_control']
                total_size = subgroup_sizes.sum()
                weighted_net_benefit = (net_benefit * subgroup_sizes).sum() / total_size if train_weighted_net_benefit > 0 else 0

                ates[i].append(weighted_net_benefit)

        pv_cates_means = [np.mean(pv_cates[i]) for i in range(len(cc))]
        pv_cates_sds = [np.std(pv_cates[i]) for i in range(len(cc))]
        pv_trees_means = [np.mean(pv_trees[i]) for i in range(len(cc))]
        pv_trees_sds = [np.std(pv_trees[i]) for i in range(len(cc))]
        ate_means = [np.mean(ates[i]) for i in range(len(cc))]
        ate_sds = [np.std(ates[i]) for i in range(len(cc))]

        for i,c in enumerate(cc):
            results_rows.append({
                'state': state,
                'c': c,
                'pv_cates_means': pv_cates_means[i],
                'pv_cates_sds': pv_cates_sds[i],
                'pv_trees_means': pv_trees_means[i],
                'pv_trees_sds': pv_trees_sds[i],
                'pv_ates_means': ate_means[i],
                'pv_ates_sds': ate_sds[i],
            })
    
    results = pd.DataFrame(results_rows)
    results.to_csv(f'results/policy_learning_results_{file_number}.csv', index=False)
