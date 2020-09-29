import argparse
import pandas as pd


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pred', type=str, default='checkpoint/Reg6.4_white_sides_auroc.pt.csv')
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

    ret = [args.pred.split('/')[-1]]

    numer, denom = get_hard_fn(merged)
    ret += ['{}/{}'.format(numer, denom)]

    numer, denom = get_hard_fn_sample(merged)
    ret += ['{}/{}'.format(numer, denom)]

    numer, denom = get_stupid_fn(merged)
    ret += ['{}/{}'.format(numer, denom)]

    numer, denom = get_stupid_fn_sample(merged)
    ret += ['{}/{}'.format(numer, denom)]

    numer, denom = get_fp(merged)
    ret += ['{}/{}'.format(numer, denom)]

    numer, denom = get_fp_sample(merged)
    ret += ['{}/{}'.format(numer, denom), '{:.4f}'.format(numer/denom)]

    ret += [args.threshold]

    return ret

def get_hard_fn(df):
    patches = df[df.sample_class == 1]
    hard_patches = patches[patches.patch_class == 1]
    fn_patches = hard_patches[hard_patches.prediction == 0]
    numer = len(fn_patches.index)
    denom = len(hard_patches.index)
    return numer, denom

def get_hard_fn_sample(df):
    patches = df[ (df.sample_class == 1) & (df.patch_class == 1) ]
    denom = len(patches.drop_duplicates(subset=['sample_id']).index)  # hard sample 총 갯수
    #numer = len(patches[patches.prediction == 0].drop_duplicates(subset=['sample_id']).index)
    numer = denom - len(patches[patches.prediction == 1].drop_duplicates(subset=['sample_id']).index)  # numer = denom - hard sample에서 나온 patch들 중 하나라도 결함인 sample의 갯수
    return numer, denom

def get_stupid_fn(df):
    patches = df[df.sample_class == 2]
    stupid_patches = patches[patches.patch_class == 1]
    fn_patches = stupid_patches[stupid_patches.prediction == 0]
    numer = len(fn_patches.index)
    denom = len(stupid_patches.index)
    return numer, denom

def get_stupid_fn_sample(df):
    patches = df[ (df.sample_class == 2) & (df.patch_class == 1) ]
    denom = len(patches.drop_duplicates(subset=['sample_id']).index)  # stupid sample 총 갯수
    #numer = len(patches[patches.prediction == 0].drop_duplicates(subset=['sample_id']).index)
    numer = denom - len(patches[patches.prediction == 1].drop_duplicates(subset=['sample_id']).index)  # numer = denom - stupid sample에서 나온 patch들 중 하나라도 결함인 sample의 갯수
    return numer, denom

def get_fp(df):
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
    #from IPython import embed; embed(); assert False


if __name__ == '__main__':
    args = get_args()
    ret = evaluate(args)
    print('model,hard_fn_patch,hard_fn_sample,stupid_fn_patch,stupid_fn_sample,fp_patch,fp_sample,fp_sample_ratio,threshold')
    print(','.join(list(map(str, ret))))
