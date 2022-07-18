from __future__ import print_function
import argparse
import h5py as h5
import os.path as path
import numpy as np
import math
import Corrfunc
from Corrfunc._countpairs import countpairs

parser = argparse.ArgumentParser()

parser.add_argument('boxsize')
parser.add_argument('bin_file')
parser.add_argument('mock_file1')
parser.add_argument('mock_file2')
parser.add_argument('output_file')
parser.add_argument('--siglnMc')
parser.add_argument('--nc')
parser.add_argument('--zphot')
parser.add_argument('--Qsel')
parser.add_argument('--delg')
parser.add_argument('--downsample_flag')

args = parser.parse_args()

boxsize = float(args.boxsize)
nthreads = 4
binfile = path.abspath(str(args.bin_file))

if args.siglnMc:
  Ncluster = math.floor( (boxsize**3.0) * float(args.nc) ) 
  infile = h5.File(str(args.mock_file1), 'r')
  halos = infile['halos']
  halos = halos[halos['mass'] > 0]
  Nh = int(len(halos))
  infile.close()
  np.random.seed(0)
  randnorms = np.random.normal(0.0, 1.0, Nh)
  percentiles = h5.File(str(args.delg), 'r')['halos']['percentile']
  halos['mass'] = np.exp( np.log(halos['mass']) + float(args.siglnMc) * randnorms + float(args.Qsel) * (percentiles - 0.5))
  halos.sort(order='mass')
  mock1 = halos[-Ncluster:]
  dummy2 = h5.File(str(args.mock_file2), 'r')
  mock2 = dummy2['particles']
else:
  dummy1 = h5.File(str(args.mock_file1), 'r')
  mock1 = dummy1['particles']rom
  dummy2 = h5.File(str(args.mock_file2), 'r')
  mock2 = dummy2['particles']
  if args.downsample_flag:
    mock1 = mock1[::10]
    mock2 = mock2[::10]
    

N1 = len(mock1)
N2 = len(mock2)

x1 = mock1['x'].astype(np.float32)
y1 = mock1['y'].astype(np.float32)

x2 = mock2['x'].astype(np.float32)
y2 = mock2['y'].astype(np.float32)

if args.zphot:
  z1 = mock1['zphot'].astype(np.float32)
  z2 = mock2['zphot'].astype(np.float32)
else:
  z1 = mock1['z'].astype(np.float32)
  z2 = mock2['z'].astype(np.float32)

results_DD = countpairs(0, nthreads, binfile, X1=x1, Y1=y1, Z1=z1, X2=x2, Y2=y2, Z2=z2, verbose=True)[0]

#print(results_DD)

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