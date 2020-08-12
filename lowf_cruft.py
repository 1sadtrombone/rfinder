import SNAPfiletools as sft
import numpy as np
import copy
import matplotlib.pyplot as plt
import datetime
from scipy.ndimage import median_filter, uniform_filter, maximum_filter, minimum_filter

data_dir = "/home/wizard/mars/data_auto_cross"
plot_dir = "/home/wizard/mars/plots/rfinder"
times_file = "/home/wizard/mars/scripts/rfinder/good_times.csv"

# rough flagging stuff
filtwin = 10
thresh = 5
max_occupancy = 0.25

day = 10

t = 950

name = f"lowf_cruft_day{day}_{filtwin}filtwin_{thresh}thresh"
pref = f"{plot_dir}/{name}"

times = np.genfromtxt(times_file)

ti = times[2*day]
tf = times[2*day+1]

time, data = sft.ctime2data(data_dir, ti, tf)

spec = data[0] # BE SURE TO LOOK AT THE POL11 STUFF TOO!!
logdata = np.log10(spec)

medfilt = median_filter(logdata, [1,filtwin])
corrected = logdata - medfilt

MAD = np.median(np.abs(corrected))

flags = (np.abs(corrected) > thresh*MAD).astype(int)

maxfilted = maximum_filter(flags,[1, filtwin]) # get rid of small gaps to the left
opened_flags = minimum_filter(maxfilted,[1, filtwin]) # bring highest flagged channel back down to where the signal really is
# now essentially filled out gaps in cruft then wiped the overhang away on the right edge (an "opening" filter)

lowest_freqs = np.argmin(opened_flags, axis=1) # find where the cruft ends

lowest_freqs = median_filter(lowest_freqs, filtwin)

occupancy = np.sum(opened_flags, axis=0) / opened_flags.shape[0]

lowest_freq = np.min(np.where(occupancy < max_occupancy))

plt.imshow(corrected, aspect='auto', interpolation='none')
plt.plot(lowest_freqs, np.arange(lowest_freqs.size), 'r')
plt.savefig(f"{pref}_corrected")

plt.figure()
plt.imshow(flags, aspect='auto', interpolation='none')
plt.plot(lowest_freqs, np.arange(lowest_freqs.size), 'r')
plt.savefig(f"{pref}_flags")

plt.show()
