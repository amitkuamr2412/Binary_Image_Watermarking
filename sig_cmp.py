from scipy.io import wavfile
import numpy as np
from numpy import *
from PIL import Image
import matplotlib.pyplot as plt
from math import floor
import pywt
[fs,sig]=wavfile.read('/home/rahul/Desktop/Sem6/EE338:-Application Assignment/Python_Codes/db2/rep_signal.wav');
# sig=sig[:,0]; 
sig=sig[0:5520];
[fs,sig_db2]=wavfile.read('/home/rahul/Desktop/Sem6/EE338:-Application Assignment/Python_Codes/db2/after_transforms.wav');
sig_db2=sig_db2[0:5520];
[fs,sig_db4]=wavfile.read('/home/rahul/Desktop/Sem6/EE338:-Application Assignment/Python_Codes/db4/after_transforms.wav');
[fs,sig_haar]=wavfile.read('/home/rahul/Desktop/Sem6/EE338:-Application Assignment/Python_Codes/haar/after_transforms.wav');
sig_db4=sig_db4[0:5520];
sig_haar=sig_haar[0:5520];
t=np.linspace(0,len(sig)-1,num=len(sig));
t=t[0:5520];

fig = plt.figure()

plt.subplot(2, 2, 1)
plt.plot(t, sig, 'm', label='Original') 
plt.legend()

plt.subplot(2, 2, 2)
plt.plot(t, sig_haar, 'r', label='Haar') 
plt.legend()

plt.subplot(2, 2, 3)
plt.plot(t, sig_db4, 'b', label="db4") 
plt.legend()

plt.subplot(2, 2, 4)
plt.plot(t, sig_db2, 'g', label='db2') 
plt.legend()

plt.savefig('Various_Signals.png')


fig = plt.figure()

plt.subplot(3, 1, 1)
plt.plot(t, sig-sig_haar, 'r', label='Original-Haar') 
plt.legend()

plt.subplot(3, 1, 2)
plt.plot(t, sig-sig_db4, 'b', label="Original-db4") 
plt.legend()

plt.subplot(3, 1, 3)
plt.plot(t, sig-sig_db2, 'g', label='Original-db2') 
plt.legend()

plt.savefig('Signal_Differences.png')


#Quantifying Signal Differences via comparison.
a=sig-sig_haar;
b=sig-sig_db2;
c=sig-sig_db4;
a=a*a;
b=b*b;
c=c*c;
a=np.mean(a);
b=np.mean(b);
c=np.mean(c);
print("Difference in signal and haar is ", a);
print("Difference in signal and db2 is ", b);
print("Difference in signal and db4 is ", c);
