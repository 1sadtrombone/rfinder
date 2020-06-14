import SNAPfiletools as sft
import numpy as np
import copy
import matplotlib.pyplot as plt
import datetime
import scipy.ndimage

import os

data_dir = f"{os.environ.get('PROJECT')}/../mars2019/auto_cross/data_auto_cross"
plot_dir = f"{os.environ.get('SCRATCH')}/rfi_plots"
times_file = 'good_times.csv'
name = "med_SVD" # string to identify plots saved with these settings
sensitivity = 7 # anything sensitivity*MAD above/below median flagged
ks_freq = 25 # size of kernel along freq axis
ks_time = 351 # size of kernel along time axis
nmode = 5 # number of SVD modes to keep

def flag(data, sensitivity):
    mediant = np.median(data, axis=0) 
    minus_medt = data - mediant
    MADt = np.median(np.abs(minus_medt), axis=0)
    # now have (freq) values to be compared to each time-dependent column

    flags = (np.abs(minus_medt) > sensitivity * MADt)
    
    return flags

times = np.genfromtxt(times_file)

ti = times[0]
tf = times[1]

time, data = sft.ctime2data(data_dir, ti, tf)

# axes are (time, freq)

subject = data[0] # BE SURE TO LOOK AT THE POL11 STUFF TOO!! (EXCEPT WHEN MIST)

# get rid of cruft below about 30MHz
startf = 300

logdata = np.log10(subject[:,startf:])

plt.title("logdata")
plt.imshow(logdata, aspect='auto')
plt.colorbar()
plt.savefig(f"{plot_dir}/{name}_logdata", dpi=600)
plt.clf()

rough_baseline = scipy.ndimage.median_filter(logdata, [ks_time, ks_freq])

rough_corrected = logdata - rough_baseline

rough_flags = flag(rough_corrected, sensitivity)

gapfilled = copy.deepcopy(rough_corrected)
gapfilled[rough_flags] = rough_baseline[rough_flags]

u, s, v = np.linalg.svd(gapfilled_logdata, 0)
first_modes = np.matmul(u[:,:first_nmode], np.matmul(np.diag(s[:first_nmode]), v[:first_nmode,:]))
minus_SVD = logdata - first_modes

baseline = scipy.ndimage.median_filter(logdata, [ks_time, ks_freq])

corrected = minus_SVD - baseline

rfi_removed = copy.deepcopy(corrected) 

flags = flag(corrected, sensitivity)

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
