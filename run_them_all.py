import SNAPfiletools as sft
import numpy as np
import rfinder
import copy
import matplotlib.pyplot as plt
import datetime
from scipy.ndimage import median_filter, minimum_filter, maximum_filter, uniform_filter

data_dir = "/home/wizard/mars/data_auto_cross"
plot_dir = "/home/wizard/mars/plots/rfinder"
times_file = "/home/wizard/mars/scripts/rfinder/good_times.csv"

# real flagging stuff
sensitivity = 3 # anything sensitivity*MAD above median flagged
med_win = 5
uni_win = [3,3]

name = f"all_good_days" # string to identify plots saved with these settings

startf = 0
stopf = 150 # until the end

#metadata = np.genfromtxt(times_file, delimiter=',')
#
#times = metadata[:,0]
#sites = metadata[:,1].astype(int)[::2]
#pols = metadata[:,2].astype(int)[::2]

days = [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 16, 17, 18, 19, 20][4:]
pols = [0]*len(days)
times = np.genfromtxt(times_file, delimiter=',')[8:]

total_occupancy = np.zeros((times.shape[0]//2, 2048))

for i in range(times.shape[0]//2):
    
    ti = times[2*i]
    tf = times[2*i+1]

    time, data = sft.ctime2data(data_dir, ti, tf)

    actual_ti = time[0]
    actual_tf = time[-1]

    subject = data[pols[i]] # BE SURE TO LOOK AT THE POL11 STUFF TOO!!

    logdata, quan_f, flattened, filtered, noisy_corrected, corrected, flags = rfinder.flagRFI(subject, intermediate_results=True)

    #logdata = 10*np.log10(subject)
    #
    #n=4;qs=np.arange(n+1)[1:-1]/n
    #quan_f = np.quantile(logdata, qs, axis=0)
    #flattened = logdata - quan_f[0]
    #
    #filtered = median_filter(flattened, [1, med_win])
    #
    #noisy_corrected = flattened - filtered
    #
    #corrected = uniform_filter(noisy_corrected, uni_win)
    #
    #MAD = np.median(np.abs(corrected - np.median(corrected)))
    #
    #flags = (corrected - np.median(corrected) > sensitivity * MAD)

    rfi_removed = np.ma.masked_where(flags, corrected)

    rfi_occ_freq = np.mean(flags, axis=0)
    rfi_occ_time = np.mean(flags, axis=1)

    total_occupancy[i] = rfi_occ_freq

    tz_correct_ti = actual_ti - 5 * 60 * 60
    start_time = datetime.datetime.utcfromtimestamp(tz_correct_ti).strftime('%x, %X')
    hours_since = (actual_tf-actual_ti) / (60*60)
    myext = [0, 125, hours_since, 0]

    plt.title("Raw Power Spectra")
    plt.imshow(logdata, aspect='auto', interpolation='none', extent=myext)
    plt.colorbar(label='dB')
    plt.xlabel('Frequency (MHz)')
    plt.ylabel(f'Hours Since {start_time}')
    plt.savefig(f"{plot_dir}/{name}_day{days[i]}_logdata", dpi=600)
    plt.clf()
    #
    ##plt.plot(np.linspace(0, 125, logdata.shape[1]), np.median(logdata, axis=0), label='median')
    ##plt.plot(np.linspace(0, 125, logdata.shape[1]), np.max(logdata, axis=0), label='maximum')
    ##plt.xlabel('Frequency (MHz)')
    ##plt.ylabel('Correlator Output (dB)')
    ##plt.legend()
    ##plt.savefig(f"{plot_dir}/{name}_site{sites[i]}_spectra", dpi=600)
    ##plt.clf()
    
    plt.title("Power Spectra with Subtracted Background")
    plt.imshow(corrected, aspect='auto', interpolation='none', extent=myext, vmin=-0.1, vmax=0.1)
    plt.colorbar(label="dB")
    plt.xlabel('Frequency (MHz)')
    plt.ylabel(f'Hours Since {start_time}')
    plt.savefig(f"{plot_dir}/{name}_day{days[i]}_corrected", dpi=600)
    plt.clf()
    
    plt.title("Flagged Power Spectra with Subtracted Background")
    plt.imshow(rfi_removed, aspect='auto', interpolation='none', extent=myext, vmin=-0.1, vmax=0.1)
    plt.colorbar(label="dB")
    plt.xlabel('Frequency (MHz)')
    plt.ylabel(f'Hours Since {start_time}')
    plt.savefig(f"{plot_dir}/{name}_day{days[i]}_rfi_removed", dpi=600)
    plt.clf()
    #
    #
    #f, ((a0, a1), (a2, a3)) = plt.subplots(2, 2, figsize=(10,8), gridspec_kw={'width_ratios': [3,1], 'height_ratios':[1,3], 'wspace':0, 'hspace':0, 'left':0.2})
    #a1.set_axis_off()
    #a0.get_xaxis().set_ticks([])
    #a3.get_yaxis().set_ticks([])
    #    
    #rfi_plot = a2.imshow(rfi_removed, aspect='auto', extent=myext, interpolation='none', vmin=-0.1, vmax=0.1)
    ##a2.plot(startf*np.ones(2), [0,hours_since], 'r')
    ##a2.plot(stopf*np.ones(2), [0,hours_since], 'r')
    #a0.plot(rfi_occ_freq*100, 'k')
    #a0.margins(0)
    #a3.plot(rfi_occ_time*100, np.arange(rfi_occ_time.size), 'k')
    #a3.margins(0)
    #a3.set_ylim(a3.get_ylim()[::-1])
    #
    #ca = f.add_axes([0.1, 0.15, 0.03, 0.5])
    #cbar = f.colorbar(rfi_plot, cax=ca)
    #cbar.set_label("dB")
    #ca.yaxis.set_label_position("left")
    #ca.yaxis.tick_left()
    #
    #a0.set_title("RFI Occupancy Using Background Subtraction Technique")
    #a2.set_ylabel(f"Hours Since {start_time}")
    #a2.set_xlabel("Frequency (MHz)")
    #plt.tight_layout()
    #
    #plt.savefig(f"{plot_dir}/{name}_day{days[i]}_occupancy", dpi=600)
    #plt.close(f)
    
overall_occ_freq = np.mean(total_occupancy, axis=0)

false_pos = 2 # percent

plt.ylim([0,30])
plt.xlim([0,125])
plt.plot(np.linspace(0,125,overall_occ_freq.size), overall_occ_freq*100, 'k')
plt.plot([0,125], false_pos*np.ones(2), 'r', label="False Flag Rate")
plt.xlabel('Frequency (MHz)')
plt.ylabel('RFI Occupancy (%)')
plt.title('RFI Occupancy Over All Time at the MARS')
plt.legend()
plt.savefig(f"{plot_dir}/{name}_occ_freq_zoom")
plt.clf()

plt.hist(overall_occ_freq*100, bins=int(np.max(overall_occ_freq*100))+1, color='black')
plt.title('RFI Occupancy Over All Time at the MARS')
plt.xlabel('RFI Occupancy (%)')
plt.ylabel('Counts')
plt.savefig(f"{plot_dir}/{name}_occ_freq_hist")
plt.clf()

plt.plot(days, np.mean(total_occupancy, axis=1)*100, 'k.')
plt.xlabel('Jul. 2019')
plt.ylabel('RFI Occupancy (%)')
plt.savefig(f"{plot_dir}/{name}_occ_time")
plt.clf()

plt.imshow(total_occupancy*100, aspect='auto', interpolation='none', extent=[0, 125, times.size, 0], vmin=false_pos, vmax=30)
plt.xlabel('Frequency (MHz)')
plt.ylabel('Days Since 5 Jul.')
plt.title('RFI Occupancy Over Time')
plt.colorbar(label='RFI Occupancy (%)')
plt.savefig(f"{plot_dir}/{name}_occ")
