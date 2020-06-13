import SNAPfiletools as sft
import numpy as np
import copy
import matplotlib.pyplot as plt
import datetime
from scipy.interpolate import RectBivariateSpline

data_dir = "/home/wizard/mars/data_auto_cross"
plot_dir = "/home/wizard/mars/plots/rfinder"
name = "spline_10MAD" # string to identify plots saved with these settings
sensitivity = 10 # anything sensitivity*MAD above/below median flagged
cs_freq = 15 # size of spline chunk along frequency axis
cs_time = 125 # size of spline chunk along time axis

ti = sft.timestamp2ctime('20190710_220700') + 3600 * 5
tf = ti + 3600 * 8

time, data = sft.ctime2data(data_dir, ti, tf)

subject = data[0] # BE SURE TO LOOK AT THE POL11 STUFF TOO!!

# get rid of cruft below about 30MHz
startf = 300

# show only this freq range in the rfi removed plot and SVD plot
plot_if = 0
plot_ff = 2000

logdata = np.log10(subject[:,startf:])

plt.title("logdata")
plt.imshow(logdata, aspect='auto')
plt.colorbar()
plt.savefig(f"{plot_dir}/{name}_logdata", dpi=600)
plt.clf()

# median over each chunk
# pad with nans, meaning if there are a few points left on the end, they get medianed over
time_padded = np.pad(logdata, ((0,(cs_time-logdata.shape[0]%cs_time)%cs_time),(0,0)), constant_values=np.nan)
time_medians = np.nanmedian(np.reshape(time_padded.T, (logdata.shape[1],-1,cs_time)), axis=2).T
# re-pad to median along the frequency axis
padded = np.pad(time_medians, ((0,0),(0,(cs_freq-time_medians.shape[1]%cs_freq)%cs_freq)), constant_values=np.nan)
# take the median, chunking along the other axis
spline_points = np.nanmedian(np.reshape(padded, (time_medians.shape[0],-1,cs_freq)), axis=2)

plt.imshow(spline_points, aspect='auto')
plt.colorbar()
plt.savefig(f"{plot_dir}/{name}_spline_points", dpi=600)
plt.clf()

# spline to get a baseline and sub it off
freq_points = np.arange(spline_points.shape[0])*cs_time+cs_time/2
time_points = np.arange(spline_points.shape[1])*cs_freq+cs_freq/2

baseline_spline = RectBivariateSpline(freq_points, time_points, spline_points)

freq_interp = np.arange(logdata.shape[0])
time_interp = np.arange(logdata.shape[1])
baseline = baseline_spline(freq_interp, time_interp)

plt.imshow(baseline, aspect='auto')
plt.colorbar()
plt.savefig(f"{plot_dir}/{name}_baseline", dpi=600)
plt.clf()

plt.plot(np.median(baseline, axis=0), label='med')
plt.plot(np.mean(baseline, axis=0), label='mean')
plt.plot(np.min(baseline, axis=0), label='min')
plt.plot(np.max(baseline, axis=0), label='max')
plt.legend()
plt.title(f"spline baseline spectrum")
plt.savefig(f"{plot_dir}/{name}_baseline_spectrum")
plt.clf()

corrected = logdata - baseline

rfi_removed = copy.deepcopy(corrected) 

flags = np.zeros_like(logdata, dtype=bool)

mediant = np.median(corrected, axis=0) 
minus_medt = corrected - mediant
MADt = np.median(np.abs(minus_medt), axis=0)
# now have (freq) values to be compared to each time-dependent column

flags = (np.abs(minus_medt) > sensitivity * MADt)

rfi_occ_freq = np.sum(flags, axis=0) / flags.shape[0]
rfi_occ_time = np.sum(flags, axis=1) / flags.shape[1]

rfi_removed = np.ma.masked_where(flags, rfi_removed)

plt.title("RFI removed")
plt.imshow(rfi_removed, aspect='auto')
plt.colorbar()
plt.savefig(f"{plot_dir}/{name}_rfi_removed", dpi=600)
plt.clf()

f, ((a0, a1), (a2, a3)) = plt.subplots(2, 2, gridspec_kw={'width_ratios': [3,1], 'height_ratios':[1,3]})
a1.set_axis_off()

a0.plot(rfi_occ_freq)
a0.margins(0)
a3.plot(rfi_occ_time, np.arange(rfi_occ_time.size))
a3.margins(0)
a3.set_ylim(a3.get_ylim()[::-1])
a2.imshow(rfi_removed, aspect='auto')

plt.title("RFI occupancy")
plt.tight_layout()
plt.savefig(f"{plot_dir}/{name}_occupancy", dpi=600)

# does the noise floor increase for narrow time bands?

variance = np.std(corrected[:,:800], axis=1)

plt.plot(variance)
plt.title("variance across freq, as fct of time")
plt.savefig(f"{plot_dir}/{name}_variance")
plt.clf()

rfi_occ_time_lowf = np.sum(flags[:,:800], axis=1) / flags[:,:800].shape[1]

plt.plot(rfi_occ_time_lowf/np.sum(rfi_occ_time_lowf), label='RFI occ.')
plt.plot(variance/np.sum(variance), label='variance')
plt.title("RFI to variance as fct of time comparison (both normed)")
plt.legend()
plt.tight_layout()
plt.savefig(f"{plot_dir}/{name}_compare_rfi_var")
plt.clf()
