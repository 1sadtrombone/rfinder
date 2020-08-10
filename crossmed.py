import SNAPfiletools as sft
import numpy as np
import copy
import matplotlib.pyplot as plt
import datetime
from scipy.ndimage import median_filter, uniform_filter, maximum_filter, minimum_filter

data_dir = "/home/wizard/mars/data_auto_cross"
plot_dir = "/home/wizard/mars/plots/rfinder"
times_file = "/home/wizard/mars/scripts/rfinder/good_times.csv"

# real flagging stuff
sensitivity = 2 # anything sensitivity*MAD above median flagged
med_win = 15
uni_win = [3,3]
n_quantiles = 3
quantile = 0

# rough flagging stuff
filtwin = 50
rough_thresh = 5
max_occupancy = 1/n_quantiles 

day = 9

name = f"crossmed_thenquantiles_unifilt_globalMAD_day{day}_{rough_thresh}roughMAD_{sensitivity}MAD_{med_win}win_{quantile}of{n_quantiles-1}quantile" # string to identify plots saved with these settings

# lowpass artifacts above 100MHz
stopf = 1638

times = np.genfromtxt(times_file)

ti = times[2*day]
tf = times[2*day+1]

t = 3500
f = 530

time, data = sft.ctime2data(data_dir, ti, tf)

spec = data[0] # BE SURE TO LOOK AT THE POL11 STUFF TOO!!

# show only this freq range in the rfi removed plot and SVD plot
plot_if = 0
plot_ff = 2100

logdata = np.log10(spec)

medfilt = median_filter(logdata, [1,filtwin])
corrected = logdata - medfilt

MAD = np.median(np.abs(corrected))

rough_flags = (np.abs(corrected) > rough_thresh*MAD).astype(int)

maxfilted = maximum_filter(rough_flags,[1, filtwin]) # get rid of small gaps to the left
opened_flags = minimum_filter(maxfilted,[1, filtwin]) # bring highest flagged channel back down to where the signal really is
# now essentially filled out gaps in cruft then wiped the overhang away on the right edge (an "opening" filter)

lowest_freqs = np.argmin(opened_flags, axis=1) # find where the cruft ends

occupancy = np.sum(opened_flags, axis=0) / opened_flags.shape[0]

lowest_freq = np.min(np.where(occupancy < max_occupancy))

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
qs = np.arange(n_quantiles)[1:]/n_quantiles
print(qs)
print(qs[quantile])

median_f = np.median(logdata, axis=0)
flattened = logdata - median_f

filtered = median_filter(flattened, [1, med_win])

quantiles = np.quantile(flattened - filtered, qs, axis=0)

noisy_corrected = flattened - filtered - quantiles[quantile]

corrected = uniform_filter(noisy_corrected, uni_win)

plt.plot(quantiles.T)
plt.figure()
plt.imshow(corrected[:,plot_if:plot_ff], aspect='auto', vmin=-0.001, vmax=0.001)
plt.colorbar()
plt.show()
exit()
plt.savefig(f"{plot_dir}/{name}_corrected", dpi=600)
plt.clf()

#plt.plot(quantiles[0,plot_if:plot_ff])
#plt.savefig(f"{plot_dir}/{name}_first_quantile")
#plt.clf()

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
plt.plot(corrected[:,f])
plt.plot(np.arange(corrected[:,f].size)[np.where(flags[:,f])], (corrected[:,f])[np.where(flags[:,f])], 'r.')
plt.plot((MAD*np.ones_like(logdata[:,500])*sensitivity))
plt.plot((MAD*np.ones_like(logdata[:,500])))

for i in range(n_quantiles-1):
    plt.plot(quantiles[i,f]*np.ones_like(logdata[:,500])-quantiles[0,f], label=i)
plt.legend()

plt.savefig(f"{plot_dir}/{name}_corrected_{f}f", dpi=600)
plt.clf()

axes = plt.gca()
axes.set_ylim([-0.005,0.005])
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
rfi_plot = a2.imshow(rfi_removed, aspect='auto', vmin=-0.001, vmax=0.001)
a2.plot(lowest_freqs, np.arange(lowest_freqs.size), 'r')
a2.plot(stopf*np.ones_like(logdata[:,500]), np.arange(logdata[:,500].size), 'r')
f.colorbar(rfi_plot)

plt.title("RFI occupancy")
plt.tight_layout()
plt.savefig(f"{plot_dir}/{name}_occupancy", dpi=600)

