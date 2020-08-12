import SNAPfiletools as sft
import numpy as np
import copy
import matplotlib.pyplot as plt
import datetime

data_dir = "/home/wizard/mars/data_auto_cross"
plot_dir = "/home/wizard/mars/plots/rfinder"
name = "lowpass" # string to identify plots saved with these settings
sensitivity = 5 # anything sensitivity*MAD above/below median flagged
lowest_modet = 1 # lowest fourier mode kept in time
lowest_modef = 1 # ''                       '' freq  
max_iters = 10

ti = sft.timestamp2ctime('20190710_220700') + 3600 * 5
tf = ti + 3600 * 8

time, data = sft.ctime2data(data_dir, ti, tf)

subject = data[0] # BE SURE TO LOOK AT THE POL11 STUFF TOO!!

# get rid of cruft below about 30MHz
start_index = 300

# show only this freq range in the rfi removed plot
plot_if = 0
plot_ff = 2000

logdata = np.log10(subject[:,start_index:])
fourier = np.fft.fft2(logdata)
fourier[lowest_modet:,lowest_modef:] = 0
corrected = np.abs(np.fft.ifft2(fourier))

plt.imshow(corrected, aspect='auto')
plt.savefig(f"{plot_dir}/{name}_corrected", dpi=600)

plt.plot(np.median(logdata, axis=0), label='med')
plt.plot(np.mean(logdata, axis=0), label='mean')
plt.plot(np.min(logdata, axis=0), label='min')
plt.plot(np.max(logdata, axis=0), label='max')
plt.legend()
plt.title(f"raw data spectrum")
plt.savefig(f"{plot_dir}/{name}_data_spectrum")
plt.clf()

rfi_removedt = copy.deepcopy(logdata) #remove in final version
rfi_removedf = copy.deepcopy(logdata) #remove in final version
rfi_removed = copy.deepcopy(logdata) 
rfi_replaced = copy.deepcopy(corrected)

flags = np.zeros_like(logdata, dtype=bool)
flagst = np.zeros_like(logdata, dtype=bool)
flagsf = np.zeros_like(logdata, dtype=bool)

last_flags = 0

for i in range(max_iters):

    """
    if i == 1:

        # second SVD, without RFI
        gapfilled_logdata = rfi_replaced + first_modes
        u, s, v = np.linalg.svd(gapfilled_logdata, 0)
        gapfilled_first_modes = np.matmul(u[:,:second_nmode], np.matmul(np.diag(s[:second_nmode]), v[:second_nmode,:]))
        rfi_replaced = gapfilled_logdata - gapfilled_first_modes
        corrected = logdata - gapfilled_first_modes

        plt.plot(np.log10(s), 'k.')
        plt.savefig(f"{plot_dir}/{name}_second_svd")
        plt.clf()

        plt.imshow(gapfilled_first_modes[:,plot_if:plot_ff], aspect='auto')
        plt.savefig(f"{plot_dir}/{name}_second_svd_firstmodes")
        plt.clf()
        
        plt.plot(np.median(gapfilled_first_modes, axis=0), label='med')
        plt.plot(np.mean(gapfilled_first_modes, axis=0), label='mean')
        plt.plot(np.min(gapfilled_first_modes, axis=0), label='min')
        plt.plot(np.max(gapfilled_first_modes, axis=0), label='max')
        plt.legend()
        plt.title(f"second SVD, first {second_nmode} modes")
        plt.savefig(f"{plot_dir}/{name}_second_svd_spectrum")
        plt.clf()

    """
            
    plt.title(f"RFI removed after {i} iterations")
    plt.imshow(rfi_removed[:,plot_if:plot_ff], aspect='auto')
    plt.colorbar()
    plt.savefig(f"{plot_dir}/{name}_rfi_iter_{i}", dpi=600)
    plt.clf()
    
    plt.title(f"gapfilled data after {i} iterations")
    plt.imshow(rfi_replaced, aspect='auto')
    plt.colorbar()
    plt.savefig(f"{plot_dir}/{name}_data_iter_{i}")
    plt.clf()

    print(f"saved figs after {i} iters")
    
    mediant = np.median(rfi_replaced, axis=0) # calculate median using replaced values
    minus_medt = corrected - mediant # here use raw data minus SVD baseline (RFI present)
    # It'll be masked out in the MAD calculation, and it'll let flagged points stay flagged between iterations
    MADt = np.ma.median(np.abs(np.ma.masked_where(flags, minus_medt)), axis=0) # use only points that haven't been flagged (yet)
    # now have (freq) values to be compared to each time-dependent column
    
    medianf = np.median(rfi_replaced, axis=1)
    minus_medf = corrected - medianf.reshape((-1,1)) # effectively a transpose
    MADf = np.ma.median(np.abs(np.ma.masked_where(flags, minus_medf)), axis=1)
    # (time) values to be compared to each freq-dependent row

    chan = 50 + plot_if
    
    plt.title(f"RFI removed in channel {chan} after {i} iterations")
    plt.plot(np.ma.masked_where(flagst, corrected)[:,chan])
    plt.plot(mediant[chan] + MADt[chan]*sensitivity*np.ones_like(rfi_replaced[:,chan]), 'C1', label="MADt")
    plt.plot(mediant[chan] - MADt[chan]*sensitivity*np.ones_like(rfi_replaced[:,chan]), 'C1')
    plt.plot(mediant[chan]*np.ones_like(rfi_replaced[:,chan]), 'C2', label="mediant")

    print(f"mediant at {chan}: {mediant[chan]}")
    print(f"MADt at {chan}: {MADt[chan]}")
    plt.legend()
    plt.savefig(f"{plot_dir}/{name}_rfi_channel_{chan}_iter_{i}")
    plt.clf()

    time = 2020

    plt.title(f"RFI removed in time {time} after {i} iterations")
    plt.plot(np.ma.masked_where(flagsf, corrected)[time])
    plt.plot(MADf[time]*sensitivity*np.ones_like(rfi_removed[time]), 'C1', label="MADf")
    plt.plot(-MADf[time]*sensitivity*np.ones_like(rfi_removed[time]), 'C1')
    plt.plot(medianf[time]*np.ones_like(rfi_removed[time]), label="medianf")
    print(f"medianf at {time}: {medianf[time]}")
    plt.legend()
    plt.savefig(f"{plot_dir}/{name}_rfi_time_{time}_iter_{i}")
    plt.clf()
    
    flagst = (np.abs(minus_medt) > sensitivity * MADt)
    flagsf = (np.abs(minus_medf) > (sensitivity * MADf).reshape((-1,1)))
    flags = flagst + flagsf
    #rfi_occ_freq = np.sum(flags, axis=0) / flags.shape[0]
    #rfi_occ_time = np.sum(flags, axis=1) / flags.shape[1]
    rfi_removedt = np.ma.masked_where(flagst, rfi_removedt)
    rfi_removedf = np.ma.masked_where(flagsf, rfi_removedf)
    rfi_removed = np.ma.masked_where(flags, rfi_removed)

    total_flags = np.sum(flags)

    print(f"new flags after {i+1} iters: {total_flags - last_flags}")

    if (total_flags - last_flags < logdata.size * (1 - 0.997)):
        print(f"converged after {i} iters")
        # I don't plot the results as the flagged points were just statistical fluctuations
        break
    
    rfi_replaced[flags] = 0 # effectively setting them to background (SVD) level

    last_flags = total_flags
        
else:
    print(f"reached max_iters={max_iters}")
