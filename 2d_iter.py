import SNAPfiletools as sft
import numpy as np
import matplotlib.pyplot as plt
import datetime

data_dir = "/home/wizard/mars/data_auto_cross"
plot_dir = "/home/wizard/mars/plots"
sensitivity = 5 # anything sensitivity*MAD above/below median flagged
nmode = 3 # number of SVD modes to subtract off
max_iters = 50

ti = 1562813825
tf = ti + 3600 * 8

time, data = sft.ctime2data(data_dir, ti, tf)

subject = data[0] # BE SURE TO LOOK AT THE POL11 STUFF TOO!!

logdata = np.log10(subject)
u, s, v = np.linalg.svd(logdata, 0)
first_modes = np.matmul(u[:,:nmode], np.matmul(np.diag(s[:nmode]), v[:nmode,:]))
corrected = logdata - first_modes

plt.plot(np.log10(s), 'k.')
plt.savefig("10th_svd")

rfi_removedt = logdata.copy()
rfi_removedf = logdata.copy()
rfi_removed = logdata.copy()

flags = np.zeros_like(logdata, dtype=bool)

last_flags = 0

for i in range(max_iters):

    plt.title(f"RFI removed after {i} iterations")
    plt.imshow(rfi_removed, aspect='auto')
    plt.colorbar()
    plt.savefig(f"{plot_dir}/10th_rfi_iter_{i}", dpi=600)
    plt.clf()
    
    plt.title(f"gapfilled data after {i} iterations")
    plt.imshow(corrected, aspect='auto')
    plt.colorbar()
    plt.savefig(f"{plot_dir}/10th_data_iter_{i}", dpi=600)
    plt.clf()
    
    mediant = np.median(corrected, axis=0) # calculate median using replaced values
    minus_medt = logdata - mediant # here use raw data
    # It'll be masked out in the MAD calculation, and it'll let flagged points stay flagged between iterations
    MADt = np.ma.median(np.abs(np.ma.masked_where(flags, minus_medt)), axis=0) # use only points that haven't been flagged (yet)
    # now have (freq) values to be compared to each time-dependent column
    
    medianf = np.median(corrected, axis=1)
    minus_medf = corrected - medianf.reshape((-1,1)) # effectively a transpose
    MADf = np.ma.median(np.abs(np.ma.masked_where(flags, minus_medf)), axis=1)
    # (time) values to be compared to each freq-dependent row
    
    flagst = (np.abs(minus_medt) > sensitivity * MADt)
    flagsf = (np.abs(minus_medf) > (sensitivity * MADf).reshape((-1,1)))
    flags = flagst + flagsf
    #rfi_occ_freq = np.sum(flags, axis=0) / flags.shape[0]
    #rfi_occ_time = np.sum(flags, axis=1) / flags.shape[1]
    rfi_removedt = np.ma.masked_where(flagst, rfi_removedt)
    rfi_removedf = np.ma.masked_where(flagsf, rfi_removedf)
    rfi_removed = np.ma.masked_where(flags, rfi_removed)

    print(np.sum(flags) - last_flags)
    if (np.sum(flags) - last_flags < logdata.size * (1 - 0.997)):
        print(f"converged after {i} iters")
        break
	
    corrected[flags] = 0 # effectively setting them to background level
    
    
else:
    print(f"reached max_iters={max_iters}")
