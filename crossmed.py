import SNAPfiletools as sft
import numpy as np
import copy
import matplotlib.pyplot as plt
import datetime
from scipy.ndimage import median_filter, uniform_filter, maximum_filter

data_dir = "/home/wizard/mars/data_auto_cross"
plot_dir = "/home/wizard/mars/plots/rfinder"
times_file = "/home/wizard/mars/scripts/rfinder/good_times.csv"

sensitivity = 5 # anything sensitivity*MAD above median flagged
med_win = 25
window = 25 # median filter window length

unifilt_win = 15 # for rough lowf cutoff
minfilt_win = 5
diff_thresh = 0.01 # mark lowest bin where -this < diff < this
occupancy_thresh = 0.25

day = 7

first_nmode = 50

name = f"crossmed_filteredmeds_globalMAD_day{day}_{sensitivity}MAD_{window}win_{med_win}medwin" # string to identify plots saved with these settings

times = np.genfromtxt(times_file)

ti = times[2*day]
tf = times[2*day+1]

t = 250
f = 530

time, data = sft.ctime2data(data_dir, ti, tf)

spec = data[0] # BE SURE TO LOOK AT THE POL11 STUFF TOO!!

# and lowpass artifacts above 100MHz
stopf = 1638

# show only this freq range in the rfi removed plot and SVD plot
plot_if = 0
plot_ff = 2100

logdata = np.log10(spec)

diff = np.diff(logdata, 1)

const_inds = (((-diff_thresh > diff) + (diff > diff_thresh)).astype(int)) # flag where diff big

outliers_out = maximum_filter(const_inds, minfilt_win) # get rid of flukes where RFI was 2-3 bins wide

lowest_freqs = np.argmin(outliers_out, axis=1)

occupancy = np.sum(outliers_out, axis=0) / outliers_out.shape[0]

lowest_freq = np.min(np.where(occupancy < occupancy_thresh))

print(lowest_freq)

plt.imshow(logdata, aspect='auto')
plt.plot(lowest_freqs, np.arange(lowest_freqs.size), 'r')
plt.figure()

plt.imshow(outliers_out, aspect='auto')
plt.plot(lowest_freqs, np.arange(lowest_freqs.size), 'r')

plt.show()
ex

plt.title("logdata")
plt.imshow(logdata[:,plot_if:plot_ff], aspect='auto')
plt.colorbar()
plt.savefig(f"{plot_dir}/{name}_logdata", dpi=600)
plt.clf()

"""
u, s, v = np.linalg.svd(logdata, 0)
first_modes = np.matmul(u[:,:first_nmode], np.matmul(np.diag(s[:first_nmode]), v[:first_nmode,:]))
corrected = logdata - first_modes

plt.imshow(corrected, aspect='auto', vmin=-.001, vmax=.001)

plt.figure()
plt.plot(np.log(s), 'k.')

plt.show()
exit()
"""

median_f = np.median(logdata, axis=0)

filtered_meds = median_filter(median_f, med_win)

flattened = logdata - filtered_meds

filtered = median_filter(flattened, [1, window])

corrected = flattened - filtered

plt.plot(np.median(corrected, axis=0))
plt.figure()

plt.imshow(corrected[:,plot_if:plot_ff], aspect='auto', vmin=-0.0025, vmax=0.0025)
plt.colorbar()
plt.savefig(f"{plot_dir}/{name}_corrected", dpi=600)
plt.clf()

plt.plot(median_f[plot_if:plot_ff])
plt.savefig(f"{plot_dir}/{name}_median_f")
plt.clf()

plt.imshow(flattened, aspect='auto')
plt.colorbar()
plt.savefig(f"{plot_dir}/{name}_flattened")
plt.clf()

MAD = np.median(np.abs(corrected))

flags = (corrected > sensitivity * MAD)

rfi_removed = np.ma.masked_where(flags, corrected)

rfi_occ_freq = np.sum(flags, axis=0) / flags.shape[0]
rfi_occ_time = np.sum(flags, axis=1) / flags.shape[1]

axes = plt.gca()
axes.set_ylim([-0.01,0.01])
plt.plot(corrected[:,f] - np.median(corrected[:,f]))
plt.plot(np.arange(corrected[:,f].size)[np.where(flags[:,f])], (corrected[:,f]-np.median(corrected[:,f]))[np.where(flags[:,f])], 'r.')
plt.plot((MAD*np.ones_like(logdata[:,500])*sensitivity))
plt.plot((MAD*np.ones_like(logdata[:,500])))
plt.savefig(f"{plot_dir}/{name}_corrected_{f}f", dpi=600)
plt.clf()

plt.plot(corrected[t])
plt.plot(np.median(corrected[t])*np.ones_like(logdata[500]))
plt.plot((MAD*sensitivity*np.ones_like(logdata[500])))
plt.savefig(f"{plot_dir}/{name}_corrected_{t}t", dpi=600)
plt.clf()
         
plt.title("RFI removed")
plt.imshow(rfi_removed[:,plot_if:plot_ff], aspect='auto', vmin=-.001, vmax=.001)
plt.colorbar()
plt.savefig(f"{plot_dir}/{name}_rfi_removed_corrected", dpi=600)
plt.clf()

plt.title("RFI removed")
plt.imshow(np.ma.masked_where(flags, logdata)[:,plot_if:plot_ff], aspect='auto')
plt.colorbar()
plt.savefig(f"{plot_dir}/{name}_rfi_removed_logdata", dpi=600)
plt.clf()

f, ((a0, a1), (a2, a3)) = plt.subplots(2, 2, gridspec_kw={'width_ratios': [3,1], 'height_ratios':[1,3]})
a1.set_axis_off()

a0.plot(rfi_occ_freq)
a0.margins(0)
a3.plot(rfi_occ_time, np.arange(rfi_occ_time.size))
a3.margins(0)
a3.set_ylim(a3.get_ylim()[::-1])
a2.imshow(rfi_removed, aspect='auto', vmin=-0.001, vmax=0.001)

plt.title("RFI occupancy")
plt.tight_layout()
plt.savefig(f"{plot_dir}/{name}_occupancy", dpi=600)

