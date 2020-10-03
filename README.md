#Acceleration of Spectral Domain Phase Microscopy for Optical Coherence Tomography
#Project for EECSE4750_001_2019_3 - HETEROGEN COMP-SIG PROCESSING
#due to the privacy of patients, we cannot share acutal data in github
This is a complete implememtation of the process from reading the raw data to computing the SDPM
Background subtractioin, windowing, resampling, FFT and SDPM are all done in GPU which achieves 20x speed up than the Matlab code.


#Prequisite

Several packages are needed to run the code

use pip install <package name> to install missing pakages

numpy 1.16.4(for data manipulation in CPU)
matplotlib 3.1.0(for plotting)
cpuy-cuda101 6.5.0(for GPU programming)

Python 3.7.3


#Data format

The input file containing data from an in vivo gerbil experiment
This is a (binary) RAW file, storing wavelength-domain detector data
Each element in the RAW file is an unsigned 16-bit integer
Each row of the RAW file is an A-scan, 2048 values corresponding to the 2048 detector pixels
There are 324 background scans, and 2048*257 data scans
This file is, as a result, several gigabytes in size. In our code, it is called:
"./TEST OLD DATA/e01r74f01a01.raw"
This data is not legal to send, as according to an NIH grant

A Hanning window of length 2048 is precomputed in MATLAB, and stored in a text file
This is called "hann.txt" and loaded in as a floating point vector

A matrix to resample wavelength to wavenumber domain is created in MATLAB. 
This matrix is a 2048-to-2048 change of basis matrix. It is, however, sparse.
The sparsity is such that about 13 "significant components" exist in each row.
We store, in "LessSparse.txt", a 2048x13 matrix containing the 13 significant components in each row.
Then, in "LessInds.txt", we store the starting indices within each row of the significant components.
We load these in as 2048x13 and 1x13 matrices respectively.

After processing, we save the processed scans in "apo.txt". Again, each row is 2048 pixels long.
There are 2048*256 rows, aswe remove the background and the first and last 1024 scans.
The 2048 pixels correspond to depth in space (spaced by 2.2 microns)
The columns correspond to time indices (spaced by 10 microseconds)


#Running the code

You should see BackgroundSub.py in the directory, it will ifrst read several files in current directory, so make sure files below are in their seats.
./TEST OLD DATA/e01r74f01a01.raw
hann.txt
LessSparse.txt
LessInds.txt

You can change the NUM_BATCH to increase or decrease the data processde per batch, according to our test, 128 is a good number.
You can change the l to change the length of the output of SDPM, the largest l is 2048, in our code, we set it to 512, since only top 512 points are needed.
After it finishes, the running time of each step will be printed out in the console, it should look like below:

It took <the time> seconds to read the data
It took <the time> seconds to obtain the polynomial background serially
It took <the time> seconds to compute SDPM
 
#Performance

![Alt text](/images/bgsub_rsp%20cpu%20and%20gpu.png?raw=true "Title")
![Alt text](/images/fft.png?raw=true "FFtT")


#Authors
BRIAN FROST
XUANYI LIAO


