import json


with open('dataset/a415f-white/insp.sample_3class_wo_invisible.json', 'r') as f:
    samples = json.loads(f.read())['insp.samples']

with open('dataset/a415f-white/single_image.2class.json', 'r') as f:
    patches = json.loads(f.read())['images']

f = open('side_annotation.csv', 'w')
f.write('patch_filename;sample_class;patch_class;coi_prediction\n')
for sample_id in sorted(samples.keys()):
    sample_c = samples[sample_id]['class'][0]

    for fname in sorted(patches[sample_id].keys()):
        if fname.split('.')[3] != 'f100':
            patch = patches[sample_id][fname]
            patch_c = patch['class'][0]
            coi_c = patch['coi_pred'][0]
            f.write('{};{};{};{}\n'.format(fname, sample_c, patch_c, coi_c))
f.close()

f = open('front_annotation.csv', 'w')
f.write('patch_filename;sample_class;patch_class;coi_prediction\n')
for sample_id in sorted(samples.keys()):
    sample_c = samples[sample_id]['class'][0]

    for fname in sorted(patches[sample_id].keys()):
        if fname.split('.')[3] == 'f100':
            patch = patches[sample_id][fname]
            patch_c = patch['class'][0]
            coi_c = patch['coi_pred'][0]
            f.write('{};{};{};{}\n'.format(fname, sample_c, patch_c, coi_c))
f.close()
