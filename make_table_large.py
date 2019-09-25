import sys
import math


if len(sys.argv) <= 5:
  print 'Usage: {} supervised_acc.csv unsupervised_acc.csv'.format(sys.argv[0])
  sys.exit(-1)

with open(sys.argv[1], 'r') as fp:
  supervised_lines = fp.readlines()
with open(sys.argv[2], 'r') as fp:
  unsupervised_lines = fp.readlines()
with open(sys.argv[3], 'r') as fp:
  supervised_lines2 = fp.readlines()
with open(sys.argv[4], 'r') as fp:
  unsupervised_lines2 = fp.readlines()

output_file = 'table.txt'
if len(sys.argv)>5:
  output_file = sys.argv[5]

assert len(supervised_lines) == len(unsupervised_lines), 'inconsistent number of rows'

def split(row):
  return [x.strip() for x in row.strip('\n ').split(',')]

lines = []
for s_row, u_row, s_row2, u_row2 in zip(supervised_lines, unsupervised_lines, supervised_lines2, unsupervised_lines2):
  s_name, s_mean, s_ci = split(s_row)
  u_name, u_mean, u_ci = split(u_row)
  s_mean, s_ci, u_mean, u_ci = float(s_mean), float(s_ci), float(u_mean), float(u_ci)
  s_name2, s_mean2, s_ci2 = split(s_row2)
  u_name2, u_mean2, u_ci2 = split(u_row2)
  s_mean2, s_ci2, u_mean2, u_ci2 = float(s_mean2), float(s_ci2), float(u_mean2), float(u_ci2)
  assert s_name2 == u_name2 and s_name2 == s_name

  # Compute CSCC mean and uncertainty
  cscc_mean = u_mean / s_mean
  cscc_ci = cscc_mean * math.sqrt((u_ci/u_mean)**2+(s_ci/s_mean)**2)

  cscc_mean2 = u_mean2 / s_mean2
  cscc_ci2 = cscc_mean2 * math.sqrt((u_ci2/u_mean2)**2+(s_ci2/s_mean2)**2)

  cells = [s_name]
  for i, elem in enumerate((s_mean, s_ci, u_mean, u_ci, cscc_mean, cscc_ci, s_mean2, s_ci2, u_mean2, u_ci2, cscc_mean2, cscc_ci2)):
    if i%2 == 0:
      cells.append('{:.2f}'.format(elem*100))
    else:
      cells.append('\\textcolor{gray}{' + '{:.2f}'.format(elem*100) + '}')
  cells_line = ' & '.join(cells) + '\\\\\\hline'
  lines.append(cells_line)

with open(output_file, 'w') as fp:
  fp.write('Supervised : {}\n'.format(sys.argv[1]))
  fp.write('Unsupervised : {}\n'.format(sys.argv[2]))
  fp.write('Supervised2 : {}\n'.format(sys.argv[1]))
  fp.write('Unsupervised2 : {}\n\n\n'.format(sys.argv[2]))
  fp.write('\n'.join(lines))

print 'Done writing at {}'.format(output_file)
