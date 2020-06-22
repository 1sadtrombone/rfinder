import SNAPfiletools as sft
import numpy as np
import copy
import matplotlib.pyplot as plt
import datetime
from scipy.ndimage import median_filter

data_dir = "/home/wizard/mars/data_auto_cross"
plot_dir = "/home/wizard/mars/plots/rfinder"
times_file = "/home/wizard/mars/scripts/rfinder/good_times.csv"
name = "param_adjust_timewise_medfilt" # string to identify plots saved with these settings

sens_min = 2
sens_max = 10
sens_step = 2
senses = np.arange(sens_min, sens_max, sens_step)

winf_min = 1
winf_max = 40
winf_step = 5
winfs = np.arange(winf_min, winf_max, winf_step)

wint_min = 1
wint_max = 40
wint_step = 5
wints = np.arange(wint_min, wint_max, wint_step)

times = np.genfromtxt(times_file)

ti = times[2]
tf = times[3]

time, data = sft.ctime2data(data_dir, ti, tf)

subject = data[0] # BE SURE TO LOOK AT THE POL11 STUFF TOO!!

# get rid of cruft below about 30MHz
startf = 300

# show only this freq range in the rfi removed plot and SVD plot
plot_if = 0
plot_ff = 2100

logdata = np.log10(subject[:,startf:])

plt.title("logdata")
plt.imshow(logdata[:,plot_if:plot_ff], aspect='auto')
plt.colorbar()
plt.savefig(f"{plot_dir}/{name}_logdata", dpi=600)
plt.clf()

for sens in senses:
    for winf in winfs:
        for wint in wints:

            filtered = median_filter(logdata, [wint, winf])

            corrected = logdata - filtered

            plt.title("corrected")
            plt.imshow(corrected[:,plot_if:plot_ff], aspect='auto')
            plt.colorbar()
            plt.savefig(f"{plot_dir}/{name}_{sens}sens_{wint}wint_{winf}winf_corrected", dpi=600)
            plt.clf()

            MAD = np.median(np.abs(corrected))

            flags = (corrected > sens * MAD)

            rfi_removed = np.ma.masked_where(flags, corrected)

            plt.title("RFI removed")
            plt.imshow(rfi_removed[:,plot_if:plot_ff], aspect='auto')
            plt.colorbar()
            plt.savefig(f"{plot_dir}/{name}_{sens}sens_{wint}wint_{winf}winf_rfi_removed", dpi=600)
            plt.clf()

            plt.title("RFI removed (logdata)")
            plt.imshow(np.ma.masked_where(flags, logdata)[:,plot_if:plot_ff], aspect='auto')
            plt.colorbar()
            plt.savefig(f"{plot_dir}/{name}_{sens}sens_{wint}wint_{winf}winf_rfi_removed_logdata", dpi=600)
            plt.clf()
