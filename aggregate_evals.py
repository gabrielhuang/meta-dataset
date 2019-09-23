import sys
import os

if len(sys.argv)!=3:
    print 'Usage: {} EXPROOT PREFIX'
EXPROOT = sys.argv[1]
PREFIX = sys.argv[2]
VALUE = 'UnsupervisedAcc_softmax'

print 'EXPROOT: {}'.format(EXPROOT)
print 'PREFIX: {}'.format(PREFIX)
out_dir = os.path.join(EXPROOT, 'eval_tables')
if not os.path.exists(out_dir):
    print 'Creating out_dir', out_dir
    os.makedirs(out_dir)
out_path = os.path.join(out_dir, 'table_{}.csv'.format(PREFIX))

datasets = [
('ilsvrc_2012','ILSVRC'),
('omniglot', 'Omniglot'),
('aircraft', 'Aircraft'),
('cu_birds', 'Birds'),
('dtd', 'Textures'),
('quickdraw', 'QuickDraw'),
('fungi', 'Fungi'),
('vgg_flower', 'VGG Flower'),
('traffic_sign', 'Traffic Sign'),
('mscoco', 'MSCOCO'),
        ]

lines = []

for name, pretty in datasets:
    # Verify dataset is there
    text_summary_path = os.path.join(EXPROOT, 'summaries', '{}_eval_{}'.format(PREFIX, name), 'test_summary.txt')
    print 'Opening {}'.format(text_summary_path)
    with open(text_summary_path, 'r') as fp:
        # Find line woit hUnsupervisedAcc_softmax
        result = None
        for line in fp:
            line = line.strip()
            if not line:
                continue
            key, val = line.split(':')
            if VALUE in key:
                mean, ci = val.split('+/-')
                mean = mean.strip().strip(',')
                ci = ci.strip().strip(',')
                result = '{},{}'.format(mean, ci)
                break
        assert val is not None, 'Failed finding {} in {}'.format(VALUE, text_summary_path)
        # Write value
        lines.append('{},{}'.format(pretty, result))

with open(out_path, 'w') as fp:
    fp.write('\n'.join(lines))
print '\n'.join(lines)
print '*'*32
print 'Wrote summary to {}'.format(out_path)

