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

day = 6

name = f"remote_survey_sites" # string to identify plots saved with these settings
print(f"writing plots called {name}")

# highpass cruft below 20MHz
# lowpass artifacts above 100MHz
startf = 20
stopf = 100

times = np.genfromtxt(times_file)

#ti = times[2*day]
#tf = times[2*day+1]

sites = np.array([4, 9, 7, 1, 8, 21, 22])
pols = np.array([0, 0, 0, 1, 1, 1, 1]) # pol to look at to get sharkie
survey_good_times = np.array(['20190710 13:32:09', '20190710 13:45:00', '20190710 14:08:28', '20190710 14:25:00', '20190710 14:56:18', '20190710 15:07:38', '20190712 11:52:00', '20190712 12:11:30', '20190712 12:34:00', '20190712 12:44:00', '20190718 12:54:48', '20190718 13:30:00', '20190718 13:45:00', '20190718 14:40:00'])

timestamps = np.zeros_like(survey_good_times, dtype=int)

for i in range(len(sites)):
    timestamps[2*i] = datetime.datetime.strptime(f'{survey_good_times[2*i]} -0500', '%Y%m%d %H:%M:%S %z').timestamp()
    timestamps[2*i+1] = datetime.datetime.strptime(f'{survey_good_times[2*i+1]} -0500', '%Y%m%d %H:%M:%S %z').timestamp()

np.savetxt('survey_times_human.csv', np.vstack((survey_good_times, np.repeat(sites,2), np.repeat(pols,2))).T, header='times, site, pol', delimiter=',', fmt='%s')
print(np.vstack((timestamps, np.repeat(sites,2), np.repeat(pols,2))).T)
np.savetxt('survey_times.csv', np.vstack((timestamps, np.repeat(sites,2), np.repeat(pols,2))).T, delimiter=',', fmt='%d')
    
exit()

t = 3500
f = 530

time, data = sft.ctime2data(data_dir, ti, tf)

actual_ti = time[0]
actual_tf = time[-1]

spec = data[pols[i]] # BE SURE TO LOOK AT THE POL11 STUFF TOO!!

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

flags = (corrected - np.median(corrected) > sensitivity * MAD)
flags_simple = (flattened > dB_thresh)

rfi_removed = np.ma.masked_where(flags, corrected)

# False Flag Rate Stuff
#print(sensitivity*MAD)
#startchan = int(startf*2048/125)
##print(np.ma.median(np.abs(rfi_removed[:,startchan:] - np.ma.median(rfi_removed[:,startchan:]))))
#no_outliers = np.ma.masked_where(np.abs(rfi_removed[:,startchan:]) > 0.05, rfi_removed[:,startchan:])
#all_rfi_removed = no_outliers.flatten()[~no_outliers.flatten().mask]
##all_rfi_removed = corrected.flatten()#[~corrected.flatten().mask]
#clean_region_rfi_removed = rfi_removed[1000:2600, 1200:1280].flatten()[~rfi_removed[1000:2600, 1200:1280].flatten().mask]
#flagged_vals = corrected[rfi_removed.mask]
#noise_sig = np.std(all_rfi_removed)
#
#M = np.max(all_rfi_removed)
#print(M)
#
#noise = np.random.normal(loc=0, scale=noise_sig, size=logdata.shape)
#
##plt.hist(rfi_removed[:,startchan:].flatten(), bins=np.linspace(-0.05,0.05))
##binned_corrected, bins, _ = plt.hist(corrected[1000:2600, 1200:1280].flatten(), bins=np.linspace(-0.01, 0.01, 100))
#binned_removed, bins, _ = plt.hist(all_rfi_removed, bins=np.linspace(-0.01, 0.01, 100))
#binned_clean_removed, _, _ = plt.hist(clean_region_rfi_removed, bins=np.linspace(-0.01, 0.01, 100))
#
##plt.figure()
##plt.plot(binned_corrected - binned_removed)
#
#ind = np.argmin((bins < -M))
#
##false_pos = (np.sum(all_rfi_removed <= 0) - np.sum((all_rfi_removed <= 0)*(all_rfi_removed > -M))) / (all_rfi_removed.size + np.sum((all_rfi_removed <= 0)*(all_rfi_removed > -M)))
##false_pos = np.sum(binned_corrected - binned_removed) / np.sum(binned_corrected)
#false_pos = np.sum(binned_removed[:ind]) / (np.sum(binned_removed) + np.sum(binned_removed[:ind]))
#false_pos_clean = np.sum(binned_clean_removed[:ind]) / (np.sum(binned_clean_removed) + np.sum(binned_clean_removed[:ind]))
#    
##print((all_rfi_removed.size + np.sum((all_rfi_removed <= 0)*(all_rfi_removed > -M))) - np.sum(binned_corrected))
##print(np.sum(binned_corrected) - corrected[1000:2600, 1200:1280].flatten().size)
#
##print(f"half: {np.sum(all_rfi_removed <= 0)}")
##print(f"within: {np.sum((all_rfi_removed <= 0)*(all_rfi_removed > -M))}")
##print(f"outside: {(np.sum(all_rfi_removed <= 0) - np.sum((all_rfi_removed <= 0)*(all_rfi_removed > -M)))}")
##print(f"total: {(all_rfi_removed.size + np.sum((all_rfi_removed <= 0)*(all_rfi_removed > -M)))}")
#print(false_pos)
#print(false_pos_clean)
#
#print(np.mean(flags[1000:2600, 1200:1280]))

rfi_occ_freq = np.mean(flags, axis=0)
rfi_occ_time = np.mean(flags, axis=1)

myext = [0, 125, actual_ti, actual_tf]

plt.title("logdata")
plt.imshow(logdata[:,plot_if:plot_ff], aspect='auto', interpolation='none', extent=myext)
plt.colorbar()
plt.savefig(f"{plot_dir}/{name}_logdata", dpi=600)
plt.show()
plt.clf()

#plt.imshow(corrected[:,plot_if:plot_ff], aspect='auto')
#plt.colorbar()
#plt.savefig(f"{plot_dir}/{name}_corrected", dpi=600)
#plt.show()
#plt.clf()

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

plt.figure()
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

rfi_plot = a2.imshow(rfi_removed, aspect='auto', extent=myext, interpolation='none', vmin=-0.1, vmax=0.1)
#a2.plot(startf*np.ones(2), [0,hours_since], 'r')
#a2.plot(stopf*np.ones(2), [0,hours_since], 'r')
a0.plot(rfi_occ_freq, 'k')
a0.margins(0)
a3.plot(rfi_occ_time, np.arange(rfi_occ_time.size), 'k')
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
