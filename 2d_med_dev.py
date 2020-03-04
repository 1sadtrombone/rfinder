import SNAPfiletools as sft
import numpy as np
import matplotlib.pyplot as plt
import datetime

data_dir = "/home/tajdyson/MARS/data_auto_cross"
sensitivity = 5 # anything this*MAD above/below median flagged

# HARDCODED data times: night of 9th
ti = 1562814420
tf = ti + 3600 * 8

time, data = sft.ctime2data(data_dir, ti, tf)

subject = data[0] # BE SURE TO LOOK AT THE POL11 STUFF TOO!!

u, s, v = np.linalg.svd(np.log10(subject), 0)
s[2:] = 0
first_two = np.matmul(u, np.matmul(np.diag(s), v))
corrected = np.log10(subject) - first_two

# make it linear again before median?

mediant = np.median(corrected, axis=0)
minus_medt = corrected - mediant
MADt = np.median(np.abs(minus_medt), axis=0)

# must transpose to interact with the right axis
medianf = np.array([np.median(corrected, axis=1)]).T
minus_medf = corrected - medianf
MADf = np.array([np.median(np.abs(minus_medf), axis=1)]).T

rfi_removedt = subject.copy()
rfi_removedf = subject.copy()
rfi_removed2d = subject.copy()
rfi_rem_min_medt = minus_medt.copy()
rfi_rem_min_medf = minus_medf.copy()
flagst = np.where(np.abs(minus_medt) > sensitivity * MADt, np.ones(subject.shape), np.zeros(subject.shape))
flagsf = np.where(np.abs(minus_medf) > sensitivity * MADf, np.ones(subject.shape), np.zeros(subject.shape))
flags2d = flagst + flagsf
#rfi_occ_freq = np.sum(flags, axis=0) / flags.shape[0]
#rfi_occ_time = np.sum(flags, axis=1) / flags.shape[1]
rfi_removedt[np.where(flagst)] = np.nan
rfi_removedf[np.where(flagsf)] = np.nan
rfi_removed2d[np.where(flags2d)] = np.nan

"""
times = [500]
for time in times:
    plt.figure()
    plt.title(time)
    plt.plot(minus_medt[time])
    plt.plot(minus_medf[time])
    plt.plot(flagsf[time])
    test_flags = np.zeros_like(subject[time])
    for i in range(test_flags.shape[0]):
        if abs(minus_medt[time,i]) > sensitivity * MADf[time]:
            test_flags[i] = 1
#    plt.plot(flagst[time])
    plt.plot([MADf[time]*sensitivity]*subject.shape[1])
    plt.plot([-MADf[time]*sensitivity]*subject.shape[1])
    plt.plot(test_flags, label='of interest')    

plt.legend()    
"""
#plt.figure()
#plt.title('minus med')
#plt.imshow(minus_med, aspect='auto', vmin=-1e-2, vmax=1e-2)
#plt.figure()
#plt.title('minus svd')
#plt.imshow(corrected, aspect='auto', vmin=-1e-2, vmax=1e-2)

plt.figure()
plt.title('rfi removed timewise, log scale')
plt.imshow(np.log10(rfi_removedt), aspect='auto', vmin=7, vmax=10)

plt.figure()
plt.title('rfi removed frequencywise, log scale')
plt.imshow(np.log10(rfi_removedf), aspect='auto', vmin=7, vmax=10)

plt.figure()
plt.title('rfi removed 2d, log scale')
plt.imshow(np.log10(rfi_removed2d), aspect='auto', vmin=7, vmax=10)

"""
#plt.figure()
#plt.title('rfi removed, minus med')
#plt.imshow(rfi_rem_min_med, aspect='auto', vmin=-1e-2, vmax=1e-2)
"""
plt.figure()
plt.title('log scale')
plt.imshow(np.log10(subject), aspect='auto', vmin=7, vmax=10)

"""

plt.figure()
index = 300
plt.title(index)
plt.plot(minus_med[:,index])
plt.plot(minus_med.shape[0]*[sensitivity*MAD[index]])
plt.plot(minus_med.shape[0]*[sensitivity*-MAD[index]])


f, ((a0, a1), (a2, a3)) = plt.subplots(2, 2, gridspec_kw={'width_ratios': [3,1], 'height_ratios':[1,3]})
a1.set_axis_off()

a0.plot(rfi_occ_freq)
a0.margins(0)
a3.plot(rfi_occ_time, np.linspace(0, len(rfi_occ_time), len(rfi_occ_time)))
a3.margins(0)
a3.set_ylim(a3.get_ylim()[::-1])
a2.imshow(np.log10(rfi_removed), aspect='auto', vmin=10, vmax=16)
plt.tight_layout()
"""
plt.show()

