import SNAPfiletools as sft
import numpy as np
import copy
import matplotlib.pyplot as plt
import datetime
from scipy.ndimage import median_filter

data_dir = "/home/wizard/mars/data_auto_cross"
plot_dir = "/home/wizard/mars/plots/rfinder"
times_file = "/home/wizard/mars/scripts/rfinder/good_times.csv"
name = "freqwise_medfilt_14MAD" # string to identify plots saved with these settings
sensitivity = 14 # anything sensitivity*MAD above/below median flagged
win = 7

times = np.genfromtxt(times_file)

ti = times[2]
tf = times[3]

time, data = sft.ctime2data(data_dir, ti, tf)

subject = data[0] # BE SURE TO LOOK AT THE POL11 STUFF TOO!!

# get rid of cruft below about 30MHz
startf = 0

# show only this freq range in the rfi removed plot and SVD plot
plot_if = 0
plot_ff = 2100

t = 250
f = 1550

logdata = np.log10(subject[:,startf:])

plt.title("logdata")
plt.imshow(logdata[:,plot_if:plot_ff], aspect='auto')
plt.colorbar()
plt.savefig(f"{plot_dir}/{name}_logdata", dpi=600)
plt.clf()

corrected = copy.deepcopy(logdata)

filtered = median_filter(corrected, size=[1,win])
corrected -= filtered

plt.imshow(corrected[:,plot_if:plot_ff], aspect='auto', vmin=-0.001, vmax=0.001)
plt.colorbar()
plt.savefig(f"{plot_dir}/{name}_corrected_{win}win", dpi=600)
plt.clf()

MAD = np.median(np.abs(corrected), axis=0)
globalMAD = np.median(np.abs(corrected))

axes = plt.gca()
axes.set_ylim([-0.01,0.01])
plt.plot(corrected[:,f])
plt.plot(np.median(corrected[:,f])*np.ones_like(logdata[:,500]))
#plt.plot((MAD[f]*sensitivity)*np.ones_like(logdata[:,500]))
plt.plot((globalMAD*sensitivity)*np.ones_like(logdata[:,500]))
plt.legend()
plt.savefig(f"{plot_dir}/{name}_corrected_{f}f", dpi=600)
plt.clf()

plt.plot(corrected[t])
plt.plot((globalMAD*sensitivity*np.ones_like(logdata[500])))
plt.legend()
plt.savefig(f"{plot_dir}/{name}_corrected_{t}t", dpi=600)
plt.clf()
         
flags = (corrected > sensitivity * globalMAD)

rfi_removed = np.ma.masked_where(flags, corrected)

rfi_occ_freq = np.sum(flags, axis=0) / flags.shape[0]
rfi_occ_time = np.sum(flags, axis=1) / flags.shape[1]

plt.title("RFI removed")
plt.imshow(rfi_removed[:,plot_if:plot_ff], aspect='auto', vmin=0)
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
a2.imshow(rfi_removed, aspect='auto', vmin=0, vmax=np.max(rfi_removed))

plt.title("RFI occupancy")
plt.tight_layout()
plt.savefig(f"{plot_dir}/{name}_occupancy", dpi=600)

