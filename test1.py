import os
import tqdm
f = open('/data1/9km/00/Z_NAFP_C_BCSH_20180101000000_P_high-warms-f03.BIN', 'rb')

for i in tqdm.tqdm(f.read()):
    print()