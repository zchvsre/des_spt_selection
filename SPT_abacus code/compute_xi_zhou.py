#script passed on from A. Salcedo 

from __future__ import print_function
import argparse
import h5py as h5
import os.path 
import numpy as np
import math
import Corrfunc
from Corrfunc._countpairs import countpairs

parser = argparse.ArgumentParser()

parser.add_argument('boxsize') #
parser.add_argument('bin_file')
parser.add_argument('mock_file1')
parser.add_argument('mock_file2')
parser.add_argument('output_file')
parser.add_argument('--siglnMc') #scatter between intrinsic 
parser.add_argument('--nc') #number density cutoff
parser.add_argument('--zphot') #photo-z
parser.add_argument('--Qsel') #assembly bias term
parser.add_argument('--delg') #galaxy-deviation from NFW
parser.add_argument('--downsample_flag')

args = parser.parse_args()

boxsize = float(args.boxsize)
nthreads = 4
binfile = os.path.abspath(str(args.bin_file))

project_path = "/global/cfs/cdirs/des/zhou/spt_lensing/"
particle_rel_path = "abacus/memHOD_11.2_12.4_0.65_1.0_0.2_0.0_0_z0p3.hdf5"
halo_rel_path = "data/halos_spt_xi_5.hdf5"
output_rel_path =  "data/testoutput"
bin_rel_path = "bin_file.txt"

particle_path = os.path.join(project_path, particle_rel_path)
halo_path = os.path.join(project_path,halo_rel_path)
output_path = os.path.join(project_path,output_rel_path)
bin_path = os.path.join(project_path,bin_rel_path)
bin_file = bin_path

boxsize = 1100
nbins = 10
rmin = 0.01
rmax = 125
bins = np.logspace(np.log10(rmin),np.log10(rmax),nbins+1)
mock_file1 = halo_path
mock_file2 = particle_path
output_file = output_path
siglnMc = 0.4
nc = 3.228E-6
Qsel = 0
nthreads = 4
zphot = False

if args.siglnMc:
    nc=3.228E-6
    Ncluster = math.floor( (boxsize**3.0) * float(nc) ) #number of cluster
    infile = h5.File(mock_file1, 'r')
    halos = infile['halos']
    halos = halos[halos['M200b'] > 0] #halo mass that passes M_min
    Nh = int(len(halos)) #number of halos
    infile.close()
    np.random.seed(0)
    randnorms = np.random.normal(0.0, 1.0, Nh)
    # percentiles = h5.File(str(args.delg), 'r')['halos']['percentile'] #Delta_gamma
    halos['Mobs'] = np.exp( np.log(halos['M200b']) + float(siglnMc) * randnorms )
                           # + float(args.Qsel) * (percentiles - 0.5))
    halos.sort(order="Mobs")
    mock1 = halos[-Ncluster:] #only choose most massive clusters
    dummy2 = h5.File(str(mock_file2), 'r')
    mock2 = dummy2['particles']
    
else:
  dummy1 = h5.File(str(args.mock_file1), 'r')
  mock1 = dummy1['particles']
  dummy2 = h5.File(str(args.mock_file2), 'r')
  mock2 = dummy2['particles']
  if args.downsample_flag:
    mock1 = mock1[::10]
    mock2 = mock2[::10]
    
    
    
N1 = len(mock1)
N2 = len(mock2)

x1 = mock1['pos_x'].astype(np.float32)
y1 = mock1['pos_y'].astype(np.float32)

x2 = mock2['x'].astype(np.float32)
y2 = mock2['y'].astype(np.float32)

if zphot:
  z1 = mock1['zphot'].astype(np.float32)
  z2 = mock2['zphot'].astype(np.float32)
else:
  z1 = mock1['pos_z'].astype(np.float32)
  z2 = mock2['z'].astype(np.float32)

        
#     input_array = [x1,y1,z1,x2,y2,z2]
    
#     for item in input_array:
#         print(item, len(item))
    
#     print(bin_file)
    
    
    
results_DD = countpairs(0, 1, bin_file, X1=x1, Y1=y1, Z1=z1, X2=x2, Y2=y2, Z2=z2, verbose=True)[0]

print(results_DD)

for i in range(0, len(results_DD)):
  RR = N1*N2*(4.0/3.0)*np.pi*(results_DD[i][1]**3.0 - results_DD[i][0]**3.0) / boxsize**3.0
  results_DD[i] = results_DD[i] + ((results_DD[i][3]/RR - 1.0),)

#print(results_DD)

outfile = open(str(args.output_file), 'w')

outfile.write("# rmin rmax npairs xi\n")

for bin in results_DD:
  outfile.write(str(bin[0])+" "+str(bin[1])+" "+str(bin[3])+" "+str(bin[5])+"\n")

outfile.close()

if args.siglnMc:
  False
else:
  dummy1.close()
dummy2.close()