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
sensitivity = 3 # anything sensitivity*MAD above median flagged
med_win = 15
uni_win = [3,3]

dB_thresh = 1

day = 10

name = f"flattened_day{day}_{dB_thresh}dB" # string to identify plots saved with these settings
print(f"writing plots called {name}")

# highpass cruft below 20MHz
# lowpass artifacts above 100MHz
startf = 20
stopf = 100

times = np.genfromtxt(times_file)

ti = times[2*day]
tf = times[2*day+1]

t = 3500
f = 530

time, data = sft.ctime2data(data_dir, ti, tf)

actual_ti = time[0]
actual_tf = time[-1]

spec = data[0] # BE SURE TO LOOK AT THE POL11 STUFF TOO!!

# show only this freq range in the rfi removed plot and SVD plot
plot_if = 0
plot_ff = 2100

logdata = 10 * np.log10(spec)

median_f = np.median(logdata, axis=0)
filtered_meds = median_filter(median_f, med_win)
flattened = logdata - filtered_meds

filtered = median_filter(flattened, [1, med_win])

noisy_corrected = flattened - filtered

corrected = uniform_filter(noisy_corrected, uni_win)

MAD = np.median(np.abs(corrected))

flags = (corrected > sensitivity * MAD)
flags_simple = (flattened > dB_thresh)

rfi_removed = np.ma.masked_where(flags, logdata)

rfi_occ_freq = np.mean(flags, axis=0)
rfi_occ_time = np.mean(flags, axis=1)

#plt.title("logdata")
#plt.imshow(logdata[:,plot_if:plot_ff], aspect='auto')
#plt.colorbar()
#plt.savefig(f"{plot_dir}/{name}_logdata", dpi=600)
#plt.clf()

plt.imshow(corrected[:,plot_if:plot_ff], aspect='auto')
plt.colorbar()
plt.savefig(f"{plot_dir}/{name}_corrected", dpi=600)
plt.show()
plt.clf()

#plt.imshow(flattened, aspect='auto')
#plt.colorbar()
#plt.savefig(f"{plot_dir}/{name}_flattened")
#plt.clf()

#axes = plt.gca()
#axes.set_ylim([-0.01,0.01])
#plt.plot(corrected[:,f])
##plt.plot(noisy_corrected[:,f])
#plt.plot(np.arange(corrected[:,f].size)[np.where(flags[:,f])], (corrected[:,f])[np.where(flags[:,f])], 'r.')
#plt.plot((MAD*np.ones_like(logdata[:,500])*sensitivity))
#plt.plot((MAD*np.ones_like(logdata[:,500])))

#plt.savefig(f"{plot_dir}/{name}_corrected_{f}f", dpi=600)
#plt.clf()
#
#axes = plt.gca()
#axes.set_ylim([-0.005,0.005])
#plt.plot(corrected[t])
#plt.plot(np.median(corrected[t])*np.ones_like(logdata[500]))
#plt.plot((MAD*sensitivity*np.ones_like(logdata[500])))
#plt.savefig(f"{plot_dir}/{name}_corrected_{t}t", dpi=600)
#plt.clf()
         
plt.title("RFI removed")
plt.imshow(rfi_removed[:,plot_if:plot_ff], aspect='auto', vmin=-.01, vmax=.01)
plt.colorbar()
plt.savefig(f"{plot_dir}/{name}_rfi_removed_corrected", dpi=600)
plt.clf()

#plt.title("RFI removed")
#plt.imshow(np.ma.masked_where(flags, logdata)[:,plot_if:plot_ff], aspect='auto')
#plt.colorbar()
#plt.savefig(f"{plot_dir}/{name}_rfi_removed_logdata", dpi=600)
#plt.clf()

f, ((a0, a1), (a2, a3)) = plt.subplots(2, 2, figsize=(10,8), gridspec_kw={'width_ratios': [3,1], 'height_ratios':[1,3], 'wspace':0, 'hspace':0, 'left':0.2})
a1.set_axis_off()
a0.get_xaxis().set_ticks([])
a3.get_yaxis().set_ticks([])

tz_correct_ti = actual_ti - 5 * 60 * 60
start_time = datetime.datetime.utcfromtimestamp(tz_correct_ti).strftime('%x, %X')
hours_since = (actual_tf-actual_ti) / (60*60)
myext = [0, 125, hours_since, 0]

rfi_plot = a2.imshow(rfi_removed, aspect='auto', extent=myext, interpolation='none')
a2.plot(startf*np.ones(2), [0,hours_since], 'r')
a2.plot(stopf*np.ones(2), [0,hours_since], 'r')
a0.plot(rfi_occ_freq)
a0.margins(0)
a3.plot(rfi_occ_time, np.arange(rfi_occ_time.size))
a3.margins(0)
a3.set_ylim(a3.get_ylim()[::-1])

ca = f.add_axes([0.1, 0.15, 0.03, 0.5])
cbar = f.colorbar(rfi_plot, cax=ca)
cbar.set_label("dB")
ca.yaxis.set_label_position("left")
ca.yaxis.tick_left()

a0.set_title("RFI occupancy")
a2.set_ylabel(f"Hours Since {start_time}")
a2.set_xlabel("Frequency [MHz]")
plt.tight_layout()

plt.savefig(f"{plot_dir}/{name}_occupancy", dpi=600)
plt.show()
