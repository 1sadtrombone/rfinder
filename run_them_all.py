import SNAPfiletools as sft
import numpy as np
import copy
import matplotlib.pyplot as plt
import datetime

data_dir = "/home/wizard/mars/data_auto_cross"
plot_dir = "/home/wizard/mars/plots/rfinder"
times_file = "good_times.csv"
name = "big_run_zoom_1100_to_1500" # string to identify plots saved with these settings
sensitivity = 5 # anything sensitivity*MAD above/below median flagged

def flag(data, sensitivity):
    mediant = np.median(data, axis=0) 
    minus_medt = data - mediant
    MADt = np.median(np.abs(minus_medt), axis=0)
    # now have (freq) values to be compared to each time-dependent column

    flags = (np.abs(minus_medt) > sensitivity * MADt)
    
    return flags

times = np.genfromtxt(times_file)

for i in range(times.shape[0]//2):
    
    ti = times[2*i]
    tf = times[2*i+1]

    time, data = sft.ctime2data(data_dir, ti, tf)

    subject = data[0] # BE SURE TO LOOK AT THE POL11 STUFF TOO!!

    # get rid of cruft below about 30MHz
    startf = 300

    # show only this freq range in the rfi removed plot and SVD plot
    plot_if = 800
    plot_ff = 1200

    logdata = np.log10(subject[:,startf:])

    plt.title("logdata")
    plt.imshow(logdata[:,plot_if:plot_ff], aspect='auto', vmax=np.min(logdata[:,plot_if:plot_ff])+0.5)
    plt.colorbar()
    plt.savefig(f"{plot_dir}/{name}_{i}_logdata", dpi=600)
    plt.clf()

    continue

    rfi_removed = copy.deepcopy(logdata) 

    flags = flag(logdata, sensitivity)

    rfi_removed = np.ma.masked_where(flags, rfi_removed)

    rfi_occ_freq = np.sum(flags, axis=0) / flags.shape[0]
    rfi_occ_time = np.sum(flags, axis=1) / flags.shape[1]


    plt.title("RFI removed")
    plt.imshow(rfi_removed[:,plot_if:plot_ff], aspect='auto')
    plt.colorbar()
    plt.savefig(f"{plot_dir}/{name}_{i}_rfi_removed", dpi=600)
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
    plt.savefig(f"{plot_dir}/{name}_{i}_occupancy", dpi=600)
    plt.clf()
