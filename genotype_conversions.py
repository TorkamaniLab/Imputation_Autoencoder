
genotype_to_int = {
    '0/0': [1, 0],
    '0|0': [1, 0],
    '0/1': [1, 1],
    '0|1': [1, 1],
    '1/0': [1, 1],
    '1|0': [1, 1],
    '1/1': [0, 1],
    '1|1': [0, 1],
    './0': [0, 0],
    './1': [0, 0],
    './.': [0, 0],
    '0/.': [0, 0],
    '1/.': [0, 0]
}

genotype_to_int_alt_signal_only = {
    '0/0': 0,
    '0|0': 0.0,
    '0/1': 1,
    '0|1': 1,
    '1/0': 1,
    '1|0': 1,
    '1/1': 2,
    '1|1': 2,
    './0': -1,
    './1': -1,
    './.': -1,
    '0/.': -1,
    '1/.': -1
}


def convert_gt_to_int(gt, alt_signal_only=False):
    if alt_signal_only is True:
        return genotype_to_int_alt_signal_only[gt[0:3]]
    return genotype_to_int[gt[0:3]]
