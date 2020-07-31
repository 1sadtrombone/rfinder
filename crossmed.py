import SNAPfiletools as sft
import numpy as np
import copy
import matplotlib.pyplot as plt
import datetime
from scipy.ndimage import median_filter

data_dir = "/home/wizard/mars/data_auto_cross"
plot_dir = "/home/wizard/mars/plots/rfinder"
times_file = "/home/wizard/mars/scripts/rfinder/good_times.csv"

sensitivity = 10 # anything sensitivity*MAD above/below median flagged
med_win = 5
window = 25 # median filter window length

name = f"crossmed_filteredmeds_anotherday_{sensitivity}MAD_{window}win_{med_win}medwin" # string to identify plots saved with these settings

times = np.genfromtxt(times_file)

ti = times[18]
tf = times[19]

time, data = sft.ctime2data(data_dir, ti, tf)

subject = data[0] # BE SURE TO LOOK AT THE POL11 STUFF TOO!!

# get rid of cruft below about 30MHz
startf = 0

# show only this freq range in the rfi removed plot and SVD plot
plot_if = 0
plot_ff = 2100

t = 250
f = 1572

logdata = np.log10(subject[:,startf:])

plt.title("logdata")
plt.imshow(logdata[:,plot_if:plot_ff], aspect='auto')
plt.colorbar()
plt.savefig(f"{plot_dir}/{name}_logdata", dpi=600)
plt.clf()

"""
n = 4
qs = np.arange(n+1)[1:-1]/(n-1)

quantiles = np.quantile(logdata, qs, axis=0)

plt.plot(quantiles[1] - quantiles[0])
plt.show()
exit()
"""

median_f = np.median(logdata, axis=0)

filtered_meds = median_filter(median_f, med_win)

minus_meds = logdata - filtered_meds

filtered = median_filter(minus_meds, [1, window])

corrected = minus_meds - filtered

plt.imshow(corrected[:,plot_if:plot_ff], aspect='auto', vmin=-0.0025, vmax=0.0025)
plt.colorbar()
plt.savefig(f"{plot_dir}/{name}_corrected", dpi=600)
plt.clf()

plt.plot(median_f[plot_if:plot_ff])
plt.savefig(f"{plot_dir}/{name}_median_f")
plt.clf()

plt.imshow(minus_meds, aspect='auto')
plt.colorbar()
plt.savefig(f"{plot_dir}/{name}_minus_meds")
plt.clf()

MAD = np.median(np.abs(corrected), axis=0)

flags = (corrected > sensitivity * MAD)

rfi_removed = np.ma.masked_where(flags, corrected)

rfi_occ_freq = np.sum(flags, axis=0) / flags.shape[0]
rfi_occ_time = np.sum(flags, axis=1) / flags.shape[1]

axes = plt.gca()
axes.set_ylim([-0.01,0.01])
plt.plot(corrected[:,f] - np.median(corrected[:,f]))
plt.plot(np.arange(corrected[:,f].size)[np.where(flags[:,f])], (corrected[:,f]-np.median(corrected[:,f]))[np.where(flags[:,f])], 'r.')
plt.plot((MAD[f]*np.ones_like(logdata[:,500])*sensitivity))
plt.plot((MAD[f]*np.ones_like(logdata[:,500])))
plt.savefig(f"{plot_dir}/{name}_corrected_{f}f", dpi=600)
plt.clf()

plt.plot(corrected[t])
plt.plot(np.median(corrected[t])*np.ones_like(logdata[500]))
plt.plot((MAD*sensitivity*np.ones_like(logdata[500])))
plt.savefig(f"{plot_dir}/{name}_corrected_{t}t", dpi=600)
plt.show()
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

