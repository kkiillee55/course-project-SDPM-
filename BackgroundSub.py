
# INCLUDES
import time

import numpy as np
import matplotlib.pyplot as plt
import cupy as cp
# Electron scaling constant (hardware constraint)
e_scale = 540

# Each raw detected signal is 2048
# There are 324 background scans
# And 2048*257 real data scans
# the sum (ROW) is the total number of rows in the input raw matrix

num_bgs = 324
raw_len = 2048
num_scans = 2048*257;
ROW = num_scans + num_bgs

# Open sample data to read in data
READ_START = time.time()
FILENAME = "./TEST OLD DATA/e01r74f01a01.raw"
FILE = open(FILENAME,"rb")
RAW_DATA = np.fromfile(FILE, dtype=np.uint16)
FILE.close()

# Load in Hanning window
FILENAME = "hann.txt"
FILE = open(FILENAME,"r")
hann = np.loadtxt(FILE, dtype=np.float32)
FILE.close()

# Load in resampling matrix (sparse)
FILENAME = "LessSparse.txt"
FILE = open(FILENAME,"r")
Msp = np.loadtxt(FILE, dtype=np.float32)
FILE.close()
Msp = Msp.reshape([2048,13])

# Load in sparse index matrix
FILENAME = "LessInds.txt"
FILE = open(FILENAME,"r")
Inds = np.loadtxt(FILE, dtype=np.int32)
FILE.close()

# Reshape the data so that each row is a sample. Scale by e_scale, as is hardware-determined
RAW_DATA = RAW_DATA.reshape(ROW,raw_len).astype(np.uint32)
RAW_DATA = RAW_DATA*e_scale
READ_TIME = time.time() - READ_START
print( "It took ", READ_TIME," seconds to read the data")

SERIAL_BG_START = time.time()
# First few scans are background
BGs = RAW_DATA[0:num_bgs,:]

#Average and normalize
avg_bg = np.mean(BGs, 0)
norm_bg = (avg_bg-np.min(avg_bg))/(np.max(avg_bg)-np.min(avg_bg))

# polynomial fit to smooth background
p = np.poly1d(np.polyfit(range(raw_len),norm_bg,9))
smooth_bg = p(range(raw_len))
SERIAL_BG_TIME = time.time() - SERIAL_BG_START
print("It took ", SERIAL_BG_TIME, " seconds to obtain the polynomial background serially")

# Compile the kernels for the whole thing
# ptdiv computes hanning window/smoothed background,
# bgsub_resamp performs background subtraction and apodizes AND resamples to k space


ptdiv_kernel=cp.RawKernel(r'''
extern "C" 
__global__ void ptdiv(float * hann, float * smooth, float * wind)
{
        // the window is defined by the pointwise division of the hanning window by the smoothed background
        // 2048 values in each, just two 1D 2048-long threadblocks. Easy.
        
        int tx = blockIdx.x*blockDim.x + threadIdx.x;

        if (tx<2048)
        {    
            wind[tx] = hann[tx]/smooth[tx];
        }
    }
        
''','ptdiv')

bgsub_kernel =cp.RawKernel(r'''
extern "C" 
__global__ void bgsub_resamp(float * in, int scansperthread, float * BG, float * window, float * Msparse, int * Inds, float * out)
{
        // Each of many 2048-long scans needs to have the whole BG subtracted from it
        // A thread can load one index of BG into local memory, and then act on that index for all scans
        // So the x index of a thread determines which BG value to load into local memory (at most 2048)
        // also determines what to multiply by from window
        // The threads themselves should act on more than one scan at a time.
        

        unsigned int tx = blockIdx.x*blockDim.x + threadIdx.x;
        unsigned int ty = blockIdx.y*blockDim.y + threadIdx.y;

        
        // Each thread loads 6*2 sparse matrix elements into shared memory (2 rows of sparse matrix elements)

        if (tx<1024)
        {    
            int start0 = Inds[tx];
            int start1 = Inds[tx+1024];

            // The matrix multiply occurs using only six indices (sparseness)
            // From Msparse, use Msparse[13*tx <= index < (6*tx)+6]
            // From in, use in[2048*idx + start <= index < 2048*idx + start + 6]

            // act on all elements at index tx from idx_0 <= i < idx_1

            int idx_0 = ty*scansperthread;
            int idx_1 = idx_0 + scansperthread;

            float temp_sum0,temp_sum1;

            for (int i = idx_0; i<idx_1; ++i)
            {
                temp_sum0 = 0; temp_sum1 = 0;
                for (int j = 0; j<13; ++j)
                {
                    
                    // Subtract background from input, then window, then matmul. Index-wise is fine
                    temp_sum0 = temp_sum0 + ((in[i*2048 + start0 + j]-BG[start0+j])*window[start0+j]*Msparse[13*tx + j]);
                    temp_sum1 = temp_sum1 + ((in[i*2048 + start1 + j]-BG[start1+j])*window[start1+j]*Msparse[13*(tx+1024) + j]);

                }
                out[i*2048 + tx] = temp_sum0;
                out[i*2048 + tx + 1024] = temp_sum1;
            
            }
        
        }
    }
''','bgsub_resamp')

SDPM_kernel = cp.RawKernel(r'''
#include <cupy/complex.cuh>
extern "C" __global__
void SDPM(complex<float>* fftres,const float k,const int l, const int batch_size, float* sdpm) {

     int tx=blockIdx.x*blockDim.x+threadIdx.x;
     int ty=blockIdx.y*blockDim.y+threadIdx.y;
     if (tx<l && ty<batch_size){
         float r= fftres[ty*l+tx].real();
         float i= fftres[ty*l+tx].imag();
         float phase=atan(i/r);
         sdpm[ty*l+tx] = k*phase;
     }
}
 ''', 'SDPM')

#mod_ptdiv = compiler.SourceModule(ptdiv_kernel)
#ptdiv = mod_ptdiv.get_function("ptdiv")
#mod_bgsub = compiler.SourceModule(bgsub_kernel)
#bgsub_resamp = mod_bgsub.get_function("bgsub_resamp")

GPU_BGSUB_START = time.time()
BG_gpu = cp.array(avg_bg,cp.float32)
smooth_gpu = cp.array(smooth_bg,cp.float32)
hann_gpu = cp.array(hann,cp.float32)
RAW_DATA = np.float32(RAW_DATA)
APO = np.zeros((num_scans-2048,2048)).astype(np.float32)
wind_gpu = cp.empty(hann.shape,cp.float32)
Msp_gpu = cp.array(Msp.reshape([Msp.size,1]),cp.float32)
Inds_gpu = cp.array(Inds,cp.int32)

NUM_BATCH = 128
BATCH_SIZE = int((num_scans-2048)/NUM_BATCH)

#ptdiv(hann_gpu,smooth_gpu,wind_gpu,block = (1024,1,1),grid = (2,1,1))
ptdiv_kernel((2,), (1024,), (hann_gpu,smooth_gpu,wind_gpu))

#change the cropping, 2048 means no cropping, change it to 512 and so you can crop it
l=512

lambda0=1310*1e-9
D=4*np.pi*1.33
k=lambda0/D
RES=np.zeros((APO.shape[0],l))

bgsub_rsp_hit=[]
cpu_hist=[]
for i in range(1,2):

    scan_batch = np.float32(RAW_DATA[(num_bgs + 1024 + BATCH_SIZE*i):(num_bgs + 1024 + BATCH_SIZE*(i+1)),:])
    scans_gpu = cp.array(scan_batch.reshape(scan_batch.size,1),cp.float32)
    APO_batch = np.float32(RAW_DATA[(num_bgs + 1024 + BATCH_SIZE*i):(num_bgs + 1024 + BATCH_SIZE*(i+1)),:])
    APO_gpu = cp.array(APO_batch.reshape(scan_batch.size,1),cp.float32)
    scansperthread = 256
    #bgsub_resamp(scans_gpu,np.int32(scansperthread),BG_gpu,wind_gpu,Msp_gpu,Inds_gpu,APO_gpu,block = (1024,1,1),grid = (1,BATCH_SIZE/scansperthread,1))



    bgsub_kernel((1,BATCH_SIZE/scansperthread,1),(1024,1,1),(scans_gpu,cp.int32(scansperthread),BG_gpu,wind_gpu,Msp_gpu,Inds_gpu,APO_gpu))
    cp.cuda.Device().synchronize()



    #######################my part##################################

    fft_in=APO_gpu.reshape(BATCH_SIZE,2048) # convert into 4096*2048
    fft_in=fft_in.astype(cp.complex64) # convert float into complex
    SDPM_batch=cp.empty((fft_in.shape[0],l),cp.float32) # output of SDPM 4096*2048
    fft_out=cp.fft.fft(fft_in, axis=1) # perform fft row-wise,each row has 2048 points

    # grid, block and arguments, output of fft, k is lambda/(4*pi*1.33), l is signal length 2048, BATCH_SIZE is 4096, SDPM_batch is output
    SDPM_kernel((int(2048/32)+1,int(BATCH_SIZE/32)+1,), (32,32,), (fft_out,cp.float32(k),cp.int32(l),cp.int32(BATCH_SIZE), SDPM_batch))
    #print(APO_gpu.shape)
    #the final output
    RES[BATCH_SIZE*i:BATCH_SIZE*(i+1),:] = SDPM_batch.get()


    APO[BATCH_SIZE*i:BATCH_SIZE*(i+1),:] = APO_gpu.get().reshape([BATCH_SIZE,2048])
GPU_BGSUB_TIME = time.time()-GPU_BGSUB_START
print ("It took ", GPU_BGSUB_TIME, " seconds to compute SDPM")

np.savetxt("apo.txt",APO[30000:31000,:])

