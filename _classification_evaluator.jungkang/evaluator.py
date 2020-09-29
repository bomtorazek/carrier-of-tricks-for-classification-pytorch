import argparse
import pandas as pd


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pred', type=str, default='checkpoint/side/Reg6.4_white_sides_auroc.pt.csv')
    parser.add_argument('--true', type=str, default='dataset/a415f-white/side_annotation.csv')
    parser.add_argument('--threshold', type=float, default=0.5)
    args = parser.parse_args()
    return args

def evaluate(args):
    preds = pd.read_csv(args.pred, sep=';')
    prediction = preds.ng_prob >= args.threshold
    prediction = prediction.map(lambda b: 1 if b else 0)
    preds.insert(3, 'prediction', prediction)
    trues = pd.read_csv(args.true, sep=';')
    merged = pd.merge(trues, preds, on='patch_filename')
    sample_id = merged.patch_filename.map(lambda fname: fname.split('.')[2])
    merged.insert(0, 'sample_id', sample_id)
    summary = get_summary(merged)
    summary['model'] = args.pred.split('/')[-1]
    summary['threshold'] = args.threshold
    return summary

def evalute_torch(filenames, ng_probs, trues_csv_path, threshold=0.5):
    """
    Args:
        filenames (list[str]): list of filenames
        ng_probs (list[float]): list of NG probabilities; its length must be
        equal to the length of filenames.
        trues_csv_path (str)
        threshold (float)
    """
    ok_probs = [ 1 - ng_prob for ng_prob in ng_probs ]
    preds = pd.DataFrame(
        list(zip(filenames, ok_probs, ng_probs)),
        columns = ['patch_filename', 'ok_prob', 'ng_prob']
    )
    prediction = preds.ng_prob >= args.threshold
    prediction = prediction.map(lambda b: 1 if b else 0)
    preds.insert(3, 'prediction', prediction)
    trues = pd.read_csv(trues_csv_path, sep=';')
    merged = pd.merge(trues, preds, on='patch_filename')
    sample_id = merged.patch_filename.map(lambda fname: fname.split('.')[2])
    merged.insert(0, 'sample_id', sample_id)
    summary = get_summary(merged)
    summary['threshold'] = args.threshold
    return summary

def get_summary(df):
    summary = {}
    numer, denom = get_hard_fn_patch(df)
    summary['hard_fn_patch'] = (numer, denom)
    numer, denom = get_hard_fn_sample(df)
    summary['hard_fn_sample'] = (numer, denom)
    numer, denom = get_stupid_fn_patch(df)
    summary['stupid_fn_patch'] = (numer, denom)
    numer, denom = get_stupid_fn_sample(df)
    summary['stupid_fn_sample'] = (numer, denom)
    numer, denom = get_fp_patch(df)
    summary['fp_patch'] = (numer, denom)
    numer, denom = get_fp_sample(df)
    summary['fp_sample'] = (numer, denom)
    return summary

def print_summary(summary):
    ret = [summary['model']]
    ret += ['{},{}'.format(*summary['hard_fn_patch'])]
    ret += ['{},{}'.format(*summary['hard_fn_sample'])]
    ret += ['{},{}'.format(*summary['stupid_fn_patch'])]
    ret += ['{},{}'.format(*summary['stupid_fn_sample'])]
    ret += ['{},{}'.format(*summary['fp_patch'])]
    ret += ['{},{}'.format(*summary['fp_sample'])]
    numer, denom = summary['fp_sample']
    ret += ['{:.4f}'.format(numer/denom)]
    ret += [summary['threshold']]
    print('model,hard_fn_patch,hard_fn_sample,stupid_fn_patch,stupid_fn_sample,fp_patch,fp_sample,fp_sample_ratio,threshold')
    print(','.join(list(map(str, ret))))

def get_hard_fn_patch(df):
    patches = df[df.sample_class == 1]
    hard_patches = patches[patches.patch_class == 1]
    fn_patches = hard_patches[hard_patches.prediction == 0]
    numer = len(fn_patches.index)
    denom = len(hard_patches.index)
    return numer, denom

def get_hard_fn_sample(df):
    patches = df[ (df.sample_class == 1) & (df.patch_class == 1) ]
    denom = len(patches.drop_duplicates(subset=['sample_id']).index)  # hard sample 총 갯수
    numer = denom - len(patches[patches.prediction == 1].drop_duplicates(subset=['sample_id']).index)  # numer = denom - hard sample에서 나온 patch들 중 하나라도 결함인 sample의 갯수
    return numer, denom

def get_stupid_fn_patch(df):
    patches = df[df.sample_class == 2]
    stupid_patches = patches[patches.patch_class == 1]
    fn_patches = stupid_patches[stupid_patches.prediction == 0]
    numer = len(fn_patches.index)
    denom = len(stupid_patches.index)
    return numer, denom

def get_stupid_fn_sample(df):
    patches = df[ (df.sample_class == 2) & (df.patch_class == 1) ]
    denom = len(patches.drop_duplicates(subset=['sample_id']).index)  # stupid sample 총 갯수
    numer = denom - len(patches[patches.prediction == 1].drop_duplicates(subset=['sample_id']).index)  # numer = denom - stupid sample에서 나온 patch들 중 하나라도 결함인 sample의 갯수
    return numer, denom

def get_fp_patch(df):
    patches = df[df.sample_class == 0]
    ok_patches = patches[patches.patch_class == 0]
    fp_patches = ok_patches[ok_patches.prediction == 1]
    numer = len(fp_patches.index)
    denom = len(ok_patches.index)
    return numer, denom

def get_fp_sample(df):
    patches = df[df.sample_class == 0]
    denom = len(patches.drop_duplicates(subset=['sample_id']).index)  # ok sample 총 갯수
    numer = len(patches[patches.prediction == 1].drop_duplicates(subset=['sample_id']).index)  # numer = ok sample에서 나온 patch들 중 하나라도 결함인 sample의 갯수
    return numer, denom

def print_summary_a415f_white(df):
    underkill = get_underkill_sample(df)
    overkill = get_overkill_sample(df)
    invisible = ['0005', '0013', '0033', '0046', '0055', '0101', '0105']
    print('sample id,underkill,overkill')
    for i in range(1, 204):
        sid = '{:04d}'.format(i)
        if sid in underkill:
            print('{},1,0'.format(sid))
        elif sid in overkill:
            print('{},0,1'.format(sid))
        elif sid in invisible:
            print('{},-,-'.format(sid))
        else:
            print('{},0,0'.format(sid))

def get_underkill_sample(df):
    true_ng_patches = df[ ( (df.sample_class == 1) | (df.sample_class == 2) ) & (df.patch_class == 1) ]
    true_ng_samples = true_ng_patches.drop_duplicates(subset=['sample_id'])
    pred_ng_patches = true_ng_patches[true_ng_patches.prediction == 1]
    pred_ng_samples = pred_ng_patches.drop_duplicates(subset=['sample_id'])
    uk_samples = set(true_ng_samples.sample_id).difference(set(pred_ng_samples.sample_id))
    return uk_samples

def get_overkill_sample(df):
    true_ok_patches = df[df.sample_class == 0]
    fp_patches = true_ok_patches[true_ok_patches.prediction == 1]
    fp_samples = fp_patches.drop_duplicates(subset=['sample_id'])
    return set(fp_samples.sample_id)


if __name__ == '__main__':
    args = get_args()
    summary = evaluate(args)
    print_summary(summary)