import SNAPfiletools as sft
import numpy as np
import copy
import matplotlib.pyplot as plt
import datetime
from scipy.ndimage import median_filter, minimum_filter, maximum_filter, uniform_filter

data_dir = "/home/wizard/mars/data_auto_cross"
plot_dir = "/home/wizard/mars/plots/rfinder"
times_file = "/home/wizard/mars/scripts/rfinder/good_times.csv"

# real flagging stuff
sensitivity = 3 # anything sensitivity*MAD above median flagged
med_win = 15
uni_win = [3,3]

# rough flagging stuff
filtwin = 50
rough_thresh = 5

name = f"big_run_crossmed_unifilt_globalMAD_{sensitivity}MAD" # string to identify plots saved with these settings

stopf = 1638

times = np.genfromtxt(times_file)

for i in range(times.shape[0]//2):
    
    ti = times[2*i]
    tf = times[2*i+1]

    time, data = sft.ctime2data(data_dir, ti, tf)

    subject = data[0] # BE SURE TO LOOK AT THE POL11 STUFF TOO!!

    # show only this freq range in the rfi removed plot and SVD plot
    plot_if = 0
    plot_ff = 2100

    logdata = np.log10(subject)

    # rough flagging for lowf cutoff
    medfilt = median_filter(logdata, [1,filtwin])
    corrected = logdata - medfilt
    
    MAD = np.median(np.abs(corrected))
    
    rough_flags = (np.abs(corrected) > rough_thresh*MAD).astype(int)
    
    maxfilted = maximum_filter(rough_flags,[1, filtwin]) # get rid of small gaps to the left
    opened_flags = minimum_filter(maxfilted,[1, filtwin]) # bring highest flagged channel back down to where the signal really is
    # now essentially filled out gaps in cruft then wiped the overhang away on the right edge (an "opening" filter)
    
    lowest_freqs = np.argmin(opened_flags, axis=1) # find where the cruft ends

    # finer flagging

    median_f = np.median(logdata, axis=0)
    flattened = logdata - median_f
    
    filtered = median_filter(flattened, [1, med_win])
    
    noisy_corrected = flattened - filtered
    
    corrected = uniform_filter(noisy_corrected, uni_win)

    plt.title("corrected")
    plt.imshow(corrected, aspect='auto', vmin=-0.001, vmax=0.001)
    plt.colorbar()
    plt.savefig(f"{plot_dir}/{name}_{i}_corrected", dpi=600)
    plt.clf()

    MAD = np.median(np.abs(corrected))

    flags = (corrected > sensitivity * MAD)

    rfi_removed = np.ma.masked_where(flags, corrected)

    rfi_occ_freq = np.sum(flags, axis=0) / flags.shape[0]
    rfi_occ_time = np.sum(flags, axis=1) / flags.shape[1]

    plt.title("RFI removed")
    plt.imshow(rfi_removed, aspect='auto', vmin=-0.001, vmax=0.001)
    plt.colorbar()
    plt.savefig(f"{plot_dir}/{name}_{i}_rfi_removed", dpi=600)
    plt.clf()

    plt.title("RFI removed (logdata)")
    plt.imshow(np.ma.masked_where(flags, logdata), aspect='auto')
    plt.colorbar()
    plt.savefig(f"{plot_dir}/{name}_{i}_rfi_removed_logdata", dpi=600)
    plt.clf()

    f, ((a0, a1), (a2, a3)) = plt.subplots(2, 2, gridspec_kw={'width_ratios': [3,1], 'height_ratios':[1,3]})
    a1.set_axis_off()

    a0.plot(rfi_occ_freq)
    a0.margins(0)
    a3.plot(rfi_occ_time, np.arange(rfi_occ_time.size))
    a3.margins(0)
    a3.set_ylim(a3.get_ylim()[::-1])
    a2.imshow(rfi_removed, aspect='auto', vmin=-0.001, vmax=0.001)
    a2.plot(lowest_freqs, np.arange(lowest_freqs.size), 'r')
    a2.plot(stopf*np.ones_like(logdata[:,500]), np.arange(logdata[:,500].size), 'r')
    
    plt.title("RFI occupancy")
    plt.tight_layout()
    plt.savefig(f"{plot_dir}/{name}_{i}_occupancy", dpi=600)
    plt.clf()
