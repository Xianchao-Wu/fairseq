import sys
import os

log='temp.log'


with open(log) as br:
    for aline in br.readlines():
        if 'python' not in aline:
            continue
        aline = aline.strip()
        aid = aline.split(' ')[0]

        acmd = 'kill {}'.format(aid)

        print(acmd)
        os.system(acmd)

