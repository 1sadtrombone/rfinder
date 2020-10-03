import SNAPfiletools as sft
import numpy as np
import copy
import matplotlib.pyplot as plt
import datetime
from scipy.ndimage import median_filter, uniform_filter, maximum_filter, minimum_filter

def flagRFI(spec, intermediate_results=False, thresh=3, med_win=5, uni_win=[3,3]):
    """
    Input: one auto/cross correlation array, like data that can be found in data_auto_cross.
    Output: boolean array of same shape as input, True where RFI exists

    intermediate_results: also return results of all intermediate steps (in order)?
    other params: adjust flagging threshold and background filter window size

    See below for an example
    """

    logdata = 10 * np.log10(spec)
    
    n = 4
    qs = np.arange(n+1)[1:-1]/n
    quan_f = np.quantile(logdata, qs, axis=0)
    
    flattened = logdata - quan_f[0]
    
    filtered = median_filter(flattened, [1, med_win])
    
    noisy_corrected = flattened - filtered
    
    corrected = uniform_filter(noisy_corrected, uni_win)
    
    MAD = np.median(np.abs(corrected - np.median(corrected)))
    
    flags = (corrected - np.median(corrected) > thresh * MAD)

    if intermediate_results:
        out = [logdata, quan_f, flattened, filtered, noisy_corrected, corrected, flags]
    else:
        out = flags
        
    return out

if __name__=='__main__':

    data_dir = "/home/wizard/mars/data_auto_cross"
    plot_dir = "/home/wizard/mars/plots/rfinder"
    times_file = "/home/wizard/mars/scripts/rfinder/good_times.csv"
    
    day = 9
    
    name = f"day{day}" # string to identify plots saved with these settings
    print(f"writing plots called {name}")
    
    # highpass cruft below 20MHz
    # lowpass artifacts above 100MHz
    
    times = np.genfromtxt(times_file)
    
    ti = times[2*day]
    tf = times[2*day+1]
    
    time, data = sft.ctime2data(data_dir, ti, tf)
    
    actual_ti = time[0]
    actual_tf = time[-1]
    
    spec = data[0] # BE SURE TO LOOK AT THE POL11 STUFF TOO!!

    results = flagRFI(spec, intermediate_results=True)

    rfi_removed = np.ma.masked_where(results[-1], results[-2])

    plt_ti = 300
    plt_tf = spec.shape[0]//3 + plt_ti

    steps = [results[0][plt_ti:plt_tf], results[2][plt_ti:plt_tf], results[5][plt_ti:plt_tf], rfi_removed[plt_ti:plt_tf]]
    ranges = [[60,120], [-0.2,0.2], [-0.01,0.01], [-0.01,0.01]]
    labels = ["Raw Power Spectra", *[f"After Step {i}" for i in [1,3,4]]] 

    tz_correct_ti = actual_ti - 5 * 60*60
    start_time = datetime.datetime.utcfromtimestamp(tz_correct_ti).strftime('%x, %X')
    hours_since = (actual_tf - actual_ti) / (60*60)

    myext = [0, 125, hours_since / 3, 0]

    plt_start_time = datetime.datetime.utcfromtimestamp(tz_correct_ti+plt_ti).strftime('%x, %X')

    fig, axs = plt.subplots(4,1)
    for i,ax in enumerate(axs):
        im = ax.imshow(steps[i], vmin=ranges[i][0], vmax=ranges[i][1], extent=myext, interpolation='none', aspect='auto')
        fig.colorbar(im, ax=ax, label='dB', aspect=7)
        ax.text(62.5,0,labels[i], horizontalalignment="center", verticalalignment="top",bbox=dict(facecolor='white', pad=2))
        ax.set_yticks([0,1,2])
        if i == 0:
            ax.set_title("The Flagging Process")
        if i < 3:
            ax.set_xticks([])
        else:
            ax.set_xlabel("Frequency (MHz)")
            
    plt.subplots_adjust(hspace=0.15)
    fig.add_subplot(111, frameon=False)
    plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
    plt.grid(False)
    plt.ylabel(f"Hours Since {plt_start_time}")
    plt.savefig(f"{plot_dir}/{name}_steps", dpi=300)
    exit()

    rfi_occ_freq = np.mean(results[-1], axis=0)
    rfi_occ_time = np.mean(results[-1], axis=1)
 
    f, ((a0, a1), (a2, a3)) = plt.subplots(2, 2, figsize=(10,8), gridspec_kw={'width_ratios': [3,1], 'height_ratios':[1,3], 'wspace':0, 'hspace':0, 'left':0.2})
    a1.set_axis_off()
    a0.get_xaxis().set_ticks([])
    a3.get_yaxis().set_ticks([])
    
    tz_correct_ti = actual_ti - 5 * 60 * 60
    start_time = datetime.datetime.utcfromtimestamp(tz_correct_ti).strftime('%x, %X')
    hours_since = (actual_tf-actual_ti) / (60*60)
    myext = [0, 125, hours_since, 0]
    
    rfi_plot = a2.imshow(rfi_removed, aspect='auto', extent=myext, interpolation='none', vmin=-0.1, vmax=0.1)
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
    
    plt.savefig(f"{plot_dir}/rfinder_example", dpi=600)


# UHH, you can ignore all below



# show only this freq range in the rfi removed plot and SVD plot
#plot_if = 0
#plot_ff = 2100
#
#plot_it = offset
#plot_ft = spec.shape[0]//3 + offset

    
#tz_correct_ti = actual_ti - 5 * 60*60
#start_time = datetime.datetime.utcfromtimestamp(tz_correct_ti).strftime('%x, %X')
#hours_since = (actual_tf - actual_ti) / (60*60)

#noise = np.random.normal(scale=MAD/0.67, size=(2000,20000))
#
#false_flag = np.mean(noise - np.median(noise) > sensitivity * MAD)

#myext = [0, 125, hours_since, 0]
#cmap = 'coolwarm'

#plt.subplot(411)
#plt.title("Raw Power Spectra")
#plt.gca().set_xticklabels([])
#plt.imshow(logdata[plot_it:plot_ft], aspect='auto', interpolation='none', extent=myext, cmap=cmap)
#plt.colorbar()
#
#plt.subplot(412)
#plt.title("\"Flattened\" Power Spectra")
#plt.gca().set_xticklabels([])
#plt.imshow(flattened[plot_it:plot_ft], aspect='auto', interpolation='none', extent=myext, vmin=-0.2, vmax=0.2, cmap=cmap)
#plt.colorbar()
#
#plt.subplot(413)
#plt.title("\"Corrected\" Power Spectra")
#plt.gca().set_xticklabels([])
#plt.imshow(corrected[plot_it:plot_ft], aspect='auto', interpolation='none', extent=myext, vmin=-0.01, vmax=0.01, cmap=cmap)
#plt.colorbar()
#
#plt.subplot(414)
#plt.title("Flagged Power Spectra")
#plt.imshow(rfi_removed[plot_it:plot_ft], aspect='auto', interpolation='none', extent=myext, vmin=-0.01, vmax=0.01, cmap=cmap)
#plt.colorbar()

#rfi_occ_freq = np.mean(flags, axis=0)
#rfi_occ_time = np.mean(flags, axis=1)

#myext = [0, 125, hours_since, 0]

#plt.title("logdata")
#plt.imshow(logdata[:,plot_if:plot_ff], aspect='auto', interpolation='none', extent=myext)
#plt.colorbar()
#plt.savefig(f"{plot_dir}/{name}_logdata", dpi=600)
#plt.clf()

#plt.imshow(corrected[:,plot_if:plot_ff], aspect='auto', vmin=-0.01, vmax=0.01, extent=myext, interpolation='none')
#plt.colorbar(label="dB")
#plt.ylabel(f"Hours Since {start_time}")
#plt.xlabel("Frequency [MHz]")
#plt.savefig(f"{plot_dir}/{name}_corrected", dpi=600)
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

#plt.figure()
#plt.imshow(rfi_removed[:,plot_if:plot_ff], aspect='auto', vmin=-.01, vmax=.01)
#plt.colorbar(label="dB")
#plt.ylabel(f"Hours Since {start_time}")
#plt.xlabel("Frequency [MHz]")
#plt.savefig(f"{plot_dir}/{name}_rfi_removed_corrected", dpi=600)
#plt.clf()

#plt.title("RFI removed")
#plt.imshow(np.ma.masked_where(flags, logdata)[:,plot_if:plot_ff], aspect='auto')
#plt.colorbar()
#plt.savefig(f"{plot_dir}/{name}_rfi_removed_logdata", dpi=600)
#plt.clf()

#f, ((a0, a1), (a2, a3)) = plt.subplots(2, 2, figsize=(10,8), gridspec_kw={'width_ratios': [3,1], 'height_ratios':[1,3], 'wspace':0, 'hspace':0, 'left':0.2})
#a1.set_axis_off()
#a0.get_xaxis().set_ticks([])
#a3.get_yaxis().set_ticks([])
#
#tz_correct_ti = actual_ti - 5 * 60 * 60
#start_time = datetime.datetime.utcfromtimestamp(tz_correct_ti).strftime('%x, %X')
#hours_since = (actual_tf-actual_ti) / (60*60)
#myext = [0, 125, hours_since, 0]
#
#rfi_plot = a2.imshow(rfi_removed, aspect='auto', extent=myext, interpolation='none', vmin=-0.1, vmax=0.1)
##a2.plot(startf*np.ones(2), [0,hours_since], 'r')
##a2.plot(stopf*np.ones(2), [0,hours_since], 'r')
#a0.plot(rfi_occ_freq, 'k')
#a0.margins(0)
#a3.plot(rfi_occ_time, np.arange(rfi_occ_time.size), 'k')
#a3.margins(0)
#a3.set_ylim(a3.get_ylim()[::-1])
#
#ca = f.add_axes([0.1, 0.15, 0.03, 0.5])
#cbar = f.colorbar(rfi_plot, cax=ca)
#cbar.set_label("dB")
#ca.yaxis.set_label_position("left")
#ca.yaxis.tick_left()
#
#a0.set_title("RFI occupancy")
#a2.set_ylabel(f"Hours Since {start_time}")
#a2.set_xlabel("Frequency [MHz]")
#plt.tight_layout()
#
#plt.savefig(f"{plot_dir}/{name}_occupancy", dpi=600)
