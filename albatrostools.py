import numpy
import struct
import datetime
#import scio

def unpack_1_bit(data, num_channels):
    real_pol0_chan0=numpy.asarray(numpy.right_shift(numpy.bitwise_and(data, 0x80), 7), dtype="int8")
    imag_pol0_chan0=numpy.asarray(numpy.right_shift(numpy.bitwise_and(data, 0x40), 6), dtype="int8")
    real_pol1_chan0=numpy.asarray(numpy.right_shift(numpy.bitwise_and(data, 0x20), 5), dtype="int8")
    imag_pol1_chan0=numpy.asarray(numpy.right_shift(numpy.bitwise_and(data, 0x10), 4), dtype="int8")
    real_pol0_chan1=numpy.asarray(numpy.right_shift(numpy.bitwise_and(data, 0x08), 3), dtype="int8")
    imag_pol0_chan1=numpy.asarray(numpy.right_shift(numpy.bitwise_and(data, 0x04), 2), dtype="int8")
    real_pol1_chan1=numpy.asarray(numpy.right_shift(numpy.bitwise_and(data, 0x02), 1), dtype="int8")
    imag_pol1_chan1=numpy.asarray(numpy.bitwise_and(data, 0x01), dtype="int8")
    real_pol0=numpy.ravel(numpy.column_stack((real_pol0_chan0, real_pol0_chan1)))
    imag_pol0=numpy.ravel(numpy.column_stack((imag_pol0_chan0, imag_pol0_chan1)))
    real_pol1=numpy.ravel(numpy.column_stack((real_pol1_chan0, real_pol1_chan1)))
    imag_pol1=numpy.ravel(numpy.column_stack((imag_pol1_chan0, imag_pol1_chan1)))
    real_pol0[real_pol0==0]=-1
    imag_pol0[imag_pol0==0]=-1
    real_pol1[real_pol1==0]=-1
    imag_pol1[imag_pol1==0]=-1
    pol0=real_pol0+1J*imag_pol0
    pol1=real_pol1+1J*imag_pol1
    del real_pol0
    del imag_pol0
    del real_pol1
    del imag_pol1
    pol0=numpy.reshape(pol0, (-1, num_channels))
    pol1=numpy.reshape(pol1, (-1, num_channels))
    return pol0, pol1
    
def unpack_2_bit(data, num_channels):
    real_pol0=numpy.asarray(numpy.right_shift(numpy.bitwise_and(data, 0xC0), 6), dtype="int8")
    imag_pol0=numpy.asarray(numpy.right_shift(numpy.bitwise_and(data, 0x30), 4), dtype="int8")
    real_pol1=numpy.asarray(numpy.right_shift(numpy.bitwise_and(data, 0x0C), 2), dtype="int8")
    imag_pol1=numpy.asarray(numpy.bitwise_and(data, 0x03), dtype="int8")
    real_pol0[real_pol0<=1]=real_pol0[real_pol0<=1]-2
    real_pol0[real_pol0>=2]=real_pol0[real_pol0>=2]-1
    imag_pol0[imag_pol0<=1]=imag_pol0[imag_pol0<=1]-2
    imag_pol0[imag_pol0>=2]=imag_pol0[imag_pol0>=2]-1
    real_pol1[real_pol1<=1]=real_pol1[real_pol1<=1]-2
    real_pol1[real_pol1>=2]=real_pol1[real_pol1>=2]-1
    imag_pol1[imag_pol1<=1]=imag_pol1[imag_pol1<=1]-2
    imag_pol1[imag_pol1>=2]=imag_pol1[imag_pol1>=2]-1
    pol0=real_pol0+1J*imag_pol0
    pol1=real_pol1+1J*imag_pol1
    del real_pol0
    del imag_pol0
    del real_pol1
    del imag_pol1
    pol0=pol0.reshape(-1, num_channels)
    pol1=pol1.reshape(-1, num_channels)
    return pol0, pol1
    
def unpack_4_bit(data, num_channels):
    pol0_bytes=data[:, 0::2]
    pol1_bytes=data[:, 1::2]
    real_pol0=numpy.asarray(numpy.right_shift(numpy.bitwise_and(pol0_bytes, 0xf0), 4), dtype="int8")
    imag_pol0=numpy.asarray(numpy.bitwise_and(pol0_bytes, 0x0f), dtype="int8")
    real_pol1=numpy.asarray(numpy.right_shift(numpy.bitwise_and(pol1_bytes, 0xf0), 4), dtype="int8")
    imag_pol1=numpy.asarray(numpy.bitwise_and(pol1_bytes, 0x0f), dtype="int8")
    real_pol0[real_pol0>8]=real_pol0[real_pol0>8]-16
    imag_pol0[imag_pol0>8]=imag_pol0[imag_pol0>8]-16
    real_pol1[real_pol1>8]=real_pol1[real_pol1>8]-16
    imag_pol1[imag_pol1>8]=imag_pol1[imag_pol1>8]-16
    pol0=real_pol0+1J*imag_pol0
    pol1=real_pol1+1J*imag_pol1
    del real_pol0
    del imag_pol0
    del real_pol1
    del imag_pol1
    pol0=pol0.reshape(-1, num_channels)
    pol1=pol1.reshape(-1, num_channels)
    return pol0, pol1

def correlate(pol0, pol1):
    pols={}
    data=[pol0, pol1]
    for i in range(2):
        for j in range(i, 2):
            pols["pol%d%d"%(i, j)]=data[i]*numpy.conj(data[j])
    return pols

def accumulate(data, acc_len):
    spectra=numpy.zeros((data.shape[0]/acc_len, data.shape[1]))
    for i in range((data.shape[0]/acc_len)-1):
        print(i)
        spectra[i]=numpy.mean(data[i*acc_len:(i+1)*acc_len], axis=0)
    return spectra

def get_header(file_name):
    file_data=open(file_name, "r")
    header_bytes=struct.unpack(">Q", file_data.read(8))[0]
    header_raw=file_data.read(header_bytes)
    header_data=numpy.frombuffer(header_raw, dtype=[("bytes_per_packet", ">Q"), ("length_channels", ">Q"), ("spectra_per_packet", ">Q"), ("bit_mode", ">Q"), ("have_trimble", ">Q"), ("channels", ">%dQ"%(int((header_bytes-80)/8))), ("gps_week", ">Q"), ("gps_seconds", ">Q"), ("gps_lat", ">d"), ("gps_lon", ">d"), ("gps_elev", ">d")])
    file_data.close()
    header={"header_bytes":8+header_bytes,
            "bytes_per_packet":header_data["bytes_per_packet"][0],
            "length_channels":header_data["length_channels"][0],
            "spectra_per_packet":header_data["spectra_per_packet"][0],
            "bit_mode":header_data["bit_mode"][0],
            "have_trimble":header_data["have_trimble"][0],
            "channels":header_data["channels"][0],
            "gps_week":header_data["gps_week"][0],
            "gps_seconds":header_data["gps_seconds"][0],
            "gps_latitude":header_data["gps_lat"][0],
            "gps_longitude":header_data["gps_lon"][0],
            "gps_elevation":header_data["gps_elev"][0]}
    if header["bit_mode"]==1:
        header["channels"]=numpy.ravel(numpy.column_stack((header["channels"], header["channels"]+1)))
        header["length_channels"]=int(header["length_channels"]*2)
    if header["bit_mode"]==4:
        header["channels"]=header["channels"][::2]
        header["length_channels"]=int(header["length_channels"]/2)
    return header

def get_data(file_name, items=-1):
    header=get_header(file_name)
    file_data=open(file_name, "r")
    file_data.seek(header["header_bytes"])
    data=numpy.fromfile(file_data, count=items, dtype=[("spec_num", ">I"), ("spectra", "%dB"%(header["bytes_per_packet"]-4))])
    file_data.close()
    if header["bit_mode"]==1:
        raw_spectra=data["spectra"].reshape(-1, header["length_channels"]/2)
        pol0, pol1=unpack_1_bit(raw_spectra, header["length_channels"])
    if header["bit_mode"]==2:
        raw_spectra=data["spectra"].reshape(-1, header["length_channels"])
        pol0, pol1=unpack_2_bit(raw_spectra, header["length_channels"])
    if header["bit_mode"]==4:
        raw_spectra=data["spectra"].reshape(-1, header["length_channels"])
        pol0, pol1=unpack_4_bit(raw_spectra, header["length_channels"])
    all_spec_num=[]
    for i in range(header["spectra_per_packet"]):
        all_spec_num.append(data["spec_num"]+i)
    spec_num=numpy.ravel(numpy.column_stack(tuple(all_spec_num)))
    return header, {"spectrum_number":spec_num, "pol0":pol0, "pol1":pol1}
