import matplotlib.pyplot as plt
import matplotlib.colors as colors
import numpy as np
import json
import sys

class MidpointNormalize(colors.Normalize):
    def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
        self.midpoint = midpoint
        colors.Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        # I'm ignoring masked values and all kinds of edge cases to make a
        # simple example...
        x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
        return np.ma.masked_array(np.interp(value, x, y))

path = sys.argv[1]
l = ['PUBLISHER_ID', 'PAGE_TLD',    'SUBDOMAIN',   'LAYOUT_ID',   'HOUR_OF_DAY', 'DAY_OF_WEEK', 'GENDER', 'AD_POSITION_ID',
'AGE_BUCKET', 'ADVERTISER_ID', 'AD_ID', 'CRRATIVE_ID', 'DEVICE_TYPE_ID', 'LINE_ID', 'USER_ID']
a = np.array(json.load(open(path)), dtype = float)
print a.shape
print a

plt.imshow(a, cmap='Reds', interpolation='nearest', norm=MidpointNormalize(midpoint=np.percentile(a, 50)))
xticks = l
plt.xticks(range(len(xticks)), xticks, size='small', rotation='vertical')
plt.yticks(range(len(xticks)), xticks, size='small')
plt.colorbar()
plt.show()

