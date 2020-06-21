import SNAPfiletools as sft
import numpy as np
import copy
import matplotlib.pyplot as plt
import datetime

data_dir = "/home/wizard/mars/data_auto_cross"
plot_dir = "/home/wizard/mars/plots/rfinder"
times_file = "/home/wizard/mars/scripts/rfinder/good_times.csv"
name = "highpass_timewise_bkgndasmed_real_noflagonnegative_globalMAD_5MAD" # string to identify plots saved with these settings
sensitivity = 5 # anything sensitivity*MAD above/below median flagged
fmode = 1500

times = np.genfromtxt(times_file)

ti = times[2]
tf = times[3]

time, data = sft.ctime2data(data_dir, ti, tf)

subject = data[0] # BE SURE TO LOOK AT THE POL11 STUFF TOO!!

# get rid of cruft below about 30MHz
startf = 300

# show only this freq range in the rfi removed plot and SVD plot
plot_if = 0
plot_ff = 2000

logdata = np.log10(subject[:,startf:])

plt.title("logdata")
plt.imshow(logdata[:,plot_if:plot_ff], aspect='auto')
plt.colorbar()
plt.savefig(f"{plot_dir}/{name}_logdata", dpi=600)
plt.clf()

fourier = np.fft.fft(logdata, axis=1)
# lowpass
fourier[:,fmode:] = 0
filtered = np.fft.ifft(fourier, axis=1)

plt.imshow(np.real(filtered[:,plot_if:plot_ff]), aspect='auto')
plt.colorbar()
plt.savefig(f"{plot_dir}/{name}_filtered")
plt.clf()

corrected = logdata - filtered

MAD = np.median(np.abs(np.real(corrected)))
# now have (freq) values to be compared to each time-dependent column

# the corrected (highpass) data is like a minus_med

plt.plot(np.real(corrected[1541]))
plt.plot((MAD*sensitivity)*np.ones_like(logdata[500]))
plt.savefig(f"{plot_dir}/{name}_corrected_1541")
plt.clf()

plt.plot(np.real(filtered)[1541])
plt.savefig(f"{plot_dir}/{name}_filt_1541")
plt.clf()

flags = (np.real(corrected) > sensitivity * MAD)

rfi_removed = np.ma.masked_where(flags, corrected)

rfi_occ_freq = np.sum(flags, axis=0) / flags.shape[0]
rfi_occ_time = np.sum(flags, axis=1) / flags.shape[1]

plt.title("RFI removed")
plt.imshow(np.abs(rfi_removed[:,plot_if:plot_ff]), aspect='auto')
plt.colorbar()
plt.savefig(f"{plot_dir}/{name}_rfi_removed_abs", dpi=600)
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
a2.imshow(np.real(rfi_removed), aspect='auto')

plt.title("RFI occupancy")
plt.tight_layout()
plt.savefig(f"{plot_dir}/{name}_occupancy", dpi=600)
