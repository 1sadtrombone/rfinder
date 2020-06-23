import SNAPfiletools as sft
import numpy as np
import copy
import matplotlib.pyplot as plt
import datetime
from scipy.ndimage import median_filter

data_dir = "/home/wizard/mars/data_auto_cross"
plot_dir = "/home/wizard/mars/plots/rfinder"
times_file = "/home/wizard/mars/scripts/rfinder/good_times.csv"
name = "param_adjust_finer" # string to identify plots saved with these settings

sens_min = 4
sens_max = 12
sens_step = 2
senses = np.arange(sens_min, sens_max, sens_step)

wint = 1

winf_min = 3
winf_max = 11
winf_step = 2
winfs = np.arange(winf_min, winf_max, winf_step)

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

t = 250

logdata = np.log10(subject[:,startf:])

plt.title("logdata")
plt.imshow(logdata[:,plot_if:plot_ff], aspect='auto')
plt.colorbar()
plt.savefig(f"{plot_dir}/{name}_logdata", dpi=600)
plt.clf()

for sens in senses:
    for winf in winfs:

            filtered = median_filter(logdata, [wint, winf])

            corrected = logdata - filtered

            plt.title("corrected")
            plt.imshow(corrected[:,plot_if:plot_ff], aspect='auto', vmax=np.max(corrected), vmin=np.min(corrected))
            plt.colorbar()
            plt.savefig(f"{plot_dir}/{name}_{sens}sens_{wint}wint_{winf}winf_corrected", dpi=600)
            plt.clf()

            MAD = np.median(np.abs(corrected))
            
            plt.plot(corrected[t])
            plt.plot((MAD*sens)*np.ones_like(logdata[500]))
            plt.savefig(f"{plot_dir}/{name}_{sens}sens_{wint}wint_{winf}winf_corrected_{t}", dpi=600)
            plt.clf()

            flags = (corrected > sens * MAD)

            rfi_removed = np.ma.masked_where(flags, corrected)

            plt.title("RFI removed")
            plt.imshow(rfi_removed[:,plot_if:plot_ff], aspect='auto', vmax=np.max(rfi_removed), vmin=np.min(rfi_removed))
            plt.colorbar()
            plt.savefig(f"{plot_dir}/{name}_{sens}sens_{wint}wint_{winf}winf_rfi_removed", dpi=600)
            plt.clf()

            plt.title("RFI removed (logdata)")
            plt.imshow(np.ma.masked_where(flags, logdata)[:,plot_if:plot_ff], aspect='auto')
            plt.colorbar()
            plt.savefig(f"{plot_dir}/{name}_{sens}sens_{wint}wint_{winf}winf_rfi_removed_logdata", dpi=600)
            plt.clf()
