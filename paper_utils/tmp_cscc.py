import math

u_mean, u_ci = 0.553, 0.005
s_mean, s_ci = 0.687, 0.005

cscc_mean = u_mean / s_mean
cscc_ci = cscc_mean * math.sqrt((u_ci / u_mean) ** 2 + (s_ci / s_mean) ** 2)

print 'CSCC {:.4f} +/- {:.4f}'.format(cscc_mean, cscc_ci)