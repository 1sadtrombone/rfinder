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

dB_thresh = 1

name = f"all_days_story" # string to identify plots saved with these settings

startf = 20
stopf = 100

times = np.genfromtxt(times_file)

total_occupancy = np.zeros((times.shape[0]//2, 2048))
total_occupancy_simple = np.zeros((times.shape[0]//2, 2048))

for i in range(times.shape[0]//2):
    
    ti = times[2*i]
    tf = times[2*i+1]

    time, data = sft.ctime2data(data_dir, ti, tf)

    actual_ti = time[0]
    actual_tf = time[-1]

    subject = data[0] # BE SURE TO LOOK AT THE POL11 STUFF TOO!!

    logdata = np.log10(subject)

    median_f = np.median(logdata, axis=0)
    filtered_meds = median_filter(median_f, med_win)
    flattened = logdata - filtered_meds
    
    filtered = median_filter(flattened, [1, med_win])
    
    noisy_corrected = flattened - filtered
    
    corrected = uniform_filter(noisy_corrected, uni_win)

    MAD = np.median(np.abs(corrected))

    flags = (corrected > sensitivity * MAD)
    flags_simple = (flattened * 10 > dB_thresh)

    rfi_removed = np.ma.masked_where(flags, corrected)
    rfi_removed_simple = np.ma.masked_where(flags_simple, logdata)

    rfi_occ_freq = np.mean(flags, axis=0)
    rfi_occ_time = np.mean(flags, axis=1)
    rfi_occ_freq_simple = np.mean(flags_simple, axis=0)
    rfi_occ_time_simple = np.mean(flags_simple, axis=1)

    total_occupancy[i] = rfi_occ_freq
    total_occupancy_simple[i] = rfi_occ_freq_simple

    tz_correct_ti = actual_ti - 5 * 60 * 60
    start_time = datetime.datetime.utcfromtimestamp(tz_correct_ti).strftime('%x, %X')
    hours_since = (actual_tf-actual_ti) / (60*60)
    myext = [0, 125, hours_since, 0]

    plt.title("Raw Power Spectra")
    plt.imshow(logdata*10, aspect='auto', interpolation='none', extent=myext)
    plt.colorbar(label='dB')
    plt.xlabel('Frequency (MHz)')
    plt.ylabel(f'Hours Since {start_time}')
    plt.savefig(f"{plot_dir}/{name}_{i}_logdata", dpi=600)
    plt.clf()

    plt.title("Power Spectra with Subtracted Background")
    plt.imshow(corrected*10, aspect='auto', interpolation='none', extent=myext, vmin=-0.1, vmax=0.1)
    plt.colorbar(label="dB")
    plt.xlabel('Frequency (MHz)')
    plt.ylabel(f'Hours Since {start_time}')
    plt.savefig(f"{plot_dir}/{name}_{i}_corrected", dpi=600)
    plt.clf()

    f, ((a0, a1), (a2, a3)) = plt.subplots(2, 2, figsize=(10,8), gridspec_kw={'width_ratios': [3,1], 'height_ratios':[1,3], 'wspace':0, 'hspace':0, 'left':0.2})
    a1.set_axis_off()
    a0.get_xaxis().set_ticks([])
    a3.get_yaxis().set_ticks([])
    
    rfi_plot = a2.imshow(rfi_removed_simple*10, aspect='auto', extent=myext, interpolation='none')
    a2.plot(startf*np.ones(2), [0,hours_since], 'r')
    a2.plot(stopf*np.ones(2), [0,hours_since], 'r')
    a0.plot(rfi_occ_freq_simple, 'k')
    a0.margins(0)
    a3.plot(rfi_occ_time_simple, np.arange(rfi_occ_time.size), 'k')
    a3.margins(0)
    a3.set_ylim(a3.get_ylim()[::-1])
    
    ca = f.add_axes([0.1, 0.15, 0.03, 0.5])
    cbar = f.colorbar(rfi_plot, cax=ca)
    cbar.set_label("dB")
    ca.yaxis.set_label_position("left")
    ca.yaxis.tick_left()
    
    a0.set_title("RFI Occupancy Using a Simple Flagging Technique")
    a2.set_ylabel(f"Hours Since {start_time}")
    a2.set_xlabel("Frequency (MHz)")
    plt.tight_layout()

    plt.savefig(f"{plot_dir}/{name}_{i}_simple_occupancy", dpi=600)
    plt.close(f)

    f, ((a0, a1), (a2, a3)) = plt.subplots(2, 2, figsize=(10,8), gridspec_kw={'width_ratios': [3,1], 'height_ratios':[1,3], 'wspace':0, 'hspace':0, 'left':0.2})
    a1.set_axis_off()
    a0.get_xaxis().set_ticks([])
    a3.get_yaxis().set_ticks([])
        
    rfi_plot = a2.imshow(rfi_removed*10, aspect='auto', extent=myext, interpolation='none', vmin=-0.1, vmax=0.1)
    a2.plot(startf*np.ones(2), [0,hours_since], 'r')
    a2.plot(stopf*np.ones(2), [0,hours_since], 'r')
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
    
    a0.set_title("RFI Occupancy Using Background Subtraction Technique")
    a2.set_ylabel(f"Hours Since {start_time}")
    a2.set_xlabel("Frequency (MHz)")
    plt.tight_layout()
    
    plt.savefig(f"{plot_dir}/{name}_{i}_occupancy", dpi=600)
    plt.close(f)
    
overall_occ_freq = np.mean(total_occupancy, axis=0)
overall_occ_freq_simple = np.mean(total_occupancy_simple, axis=0)

plt.plot([0,125], 0.02*np.ones(2), 'r', label="False Flag Rate")
plt.plot(np.linspace(0,125,overall_occ_freq.size), overall_occ_freq, 'k')
plt.xlabel('Frequency (MHz)')
plt.ylabel('RFI Occupancy')
plt.title('RFI Occupancy Over All Time Using Background Subtraction Technique')
plt.legend()
plt.savefig(f"{plot_dir}/{name}_occ_freq")
plt.clf()

plt.plot(np.linspace(0,125,overall_occ_freq.size), overall_occ_freq_simple, 'k')
plt.xlabel('Frequency (MHz)')
plt.ylabel('RFI Occupancy')
plt.title('RFI Occupancy Over All Time Using Simple Flagging Technique')
plt.savefig(f"{plot_dir}/{name}_occ_freq_simple")
