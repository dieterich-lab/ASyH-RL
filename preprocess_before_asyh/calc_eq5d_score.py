#!/usb/bin/env python
# the following script compute EQ5D-5L score based on the following STATA script https://euroqol-domain.ams3.digitaloceanspaces.com/wp-content/uploads/2025/07/18154132/Germany-5L-Ludwig-Hybrid-2018.txt
# https://euroqol.org/information-and-support/resources/value-sets/

import numpy as np

# calculate the EQ-5D-5L score
def calc_eq5d_score(idx_mo, idx_sc, idx_ua, idx_pd, idx_ad):
    mo = convert_mo_indices_to_nums(idx_mo)
    sc = convert_sc_indices_to_nums(idx_sc)
    ua = convert_ua_indices_to_nums(idx_ua)
    pd = convert_pd_indices_to_nums(idx_pd)
    ad = convert_ad_indices_to_nums(idx_ad)
    total = mo + sc + ua + pd + ad
    EQ = max(1 - total, 0)
    return round(EQ, 3)


# convert mobility values into float numbers
def convert_mo_indices_to_nums(idx_mo):
    # import ipdb; ipdb.set_trace()
    if isinstance(idx_mo, float) and not np.isnan(idx_mo):
        idx_mo = int(idx_mo)
    else:
        return 0.5
    
    # assert idx_mo in [1, 2, 3, 4, 5], 'mobility index has invalid value!'
    match idx_mo:
        case 1:
            return 0
        case 2:
            return 0.026
        case 3:
            return 0.042
        case 4:
            return 0.139
        case 5:
            return 0.224
        case _:
            return 0.5


# convert self-care values into float numbers
def convert_sc_indices_to_nums(idx_sc):
    # idx_sc = int(idx_sc)
    if isinstance(idx_sc, float) and not np.isnan(idx_sc):
        idx_sc = int(idx_sc)
    else:
        return 0.5
    
    # assert idx_sc in [1, 2, 3, 4, 5], 'sc index has invalid value!'
    match idx_sc:
        case 1:
            return 0
        case 2:
            return 0.050
        case 3:
            return 0.056
        case 4:
            return 0.169
        case 5:
            return 0.260
        case _:
            return 0.5


# convert daily activity values
def convert_ua_indices_to_nums(idx_ua):
    # idx_ua = int(idx_ua)
    if isinstance(idx_ua, float) and not np.isnan(idx_ua):
        idx_ua = int(idx_ua)
    else:
        return 0.5

    # assert idx_ua in [1, 2, 3, 4, 5], 'ua index has invalid value!'
    match idx_ua:
        case 1:
            return 0
        case 2:
            return 0.036
        case 3:
            return 0.049
        case 4:
            return 0.129
        case 5:
            return 0.209
        case _:
            return 0.5


# convert pain and somatic symptom values
def convert_pd_indices_to_nums(idx_pd):
    # idx_pd = int(idx_pd)
    if isinstance(idx_pd, float) and not np.isnan(idx_pd):
        idx_pd = int(idx_pd)
    else:
        return 1.0
    
    # assert idx_pd in [1, 2, 3, 4, 5], 'pd index has invalid value!'
    match idx_pd:
        case 1:
            return 0
        case 2:
            return 0.057
        case 3:
            return 0.109
        case 4:
            return 0.404
        case 5:
            return 0.612
        case _:
            return 1.0

# convert anxiety and depression values into float numbers
def convert_ad_indices_to_nums(idx_ad):
    # idx_ad = int(idx_ad)
    if isinstance(idx_ad, float) and not np.isnan(idx_ad):
        idx_ad = int(idx_ad)
    else:
        return 0.7
    
    # if idx_ad not in list(range(6)):
    #     import ipdb; ipdb.set_trace()

    # assert idx_ad in [1, 2, 3, 4, 5], 'ad index has invalid value!'
    match idx_ad:
        case 1:
            return 0
        case 2:
            return 0.030
        case 3:
            return 0.082
        case 4:
            return 0.244
        case 5:
            return 0.356
        case _:
            return 0.7
