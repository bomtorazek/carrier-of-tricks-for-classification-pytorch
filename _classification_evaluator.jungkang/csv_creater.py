import json


with open('dataset/a415f-white/insp.sample_3class_wo_invisible.json', 'r') as f:
    samples = json.loads(f.read())['insp.samples']

with open('dataset/a415f-white/single_image.2class.json', 'r') as f:
    patches = json.loads(f.read())['images']

f = open('side_annotation.csv', 'w')
f.write('patch_filename;sample_class;patch_class\n')
for sample_id in sorted(samples.keys()):
    sample_class = samples[sample_id]['class'][0]
    if sample_class in [1, 2]:
        if 1 not in samples[sample_id]['surfaces'] and 2 not in samples[sample_id]['surfaces']:
            sample_class = 0

    for fname in sorted(patches[sample_id].keys()):
        if fname.split('.')[3] != 'f100':
            patch = patches[sample_id][fname]
            patch_class = patch['class'][0]
            f.write('{};{};{}\n'.format(fname, sample_class, patch_class))
f.close()

f = open('front_annotation.csv', 'w')
f.write('patch_filename;sample_class;patch_class\n')
for sample_id in sorted(samples.keys()):
    sample_class = samples[sample_id]['class'][0]
    if sample_class in [1, 2]:
        if 0 not in samples[sample_id]['surfaces']:
            sample_class = 0

    for fname in sorted(patches[sample_id].keys()):
        if fname.split('.')[3] == 'f100':
            patch = patches[sample_id][fname]
            patch_class = patch['class'][0]
            f.write('{};{};{}\n'.format(fname, sample_class, patch_class))
f.close()
