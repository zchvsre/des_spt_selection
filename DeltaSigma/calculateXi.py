import h5py as h5
import os.path
import numpy as np
import math
import Corrfunc
from Corrfunc._countpairs import countpairs 

project_path = "/global/cfs/cdirs/des/zhou/spt_lensing/"
particle_rel_path = "abacus/memHOD_11.2_12.4_0.65_1.0_0.2_0.0_0_z0p3.hdf5"
halo_rel_path = "data/halos_spt_xi_5.pkl"
output_rel_path =  "data/testoutput"

particle_path = os.path.join(project_path, particle_rel_path)
halo_path = os.path.join(project_path,halo_rel_path)
output_path = os.path.join(project_path,output_rel_path)

boxsize = 1100
bins = np.exp(np.linspace(np.log(0.05),np.log(125),100))
mock_file1 = halo_path
mock_file2 = particle_path
output_file = output_path
siglnMc = 0.4
nc = 3.228E-6
Qsel = 0
nthreads = 4

def writeXi(boxsize,bins,mock_file1,mock_file2,output_file,siglnMc,nc,Qsel,nthreads): 
  Ncluster = math.floor( (boxsize**3.0) * float(nc) ) #number of cluster
  infile = h5.File(str(mock_file1), 'r')
  halos = infile['halos'] 
  halos = halos[halos['M200b'] > 0] #halo mass that passes M_min
  Nh = int(len(halos)) #number of halos
  infile.close()
  np.random.seed(0)
  randnorms = np.random.normal(0.0, 1.0, Nh)
  # percentiles = h5.File(str(args.delg), 'r')['halos']['percentile'] #Delta_gamma
  halos['mass'] = np.exp( np.log(halos['mass']) + float(siglnMc) * randnorms )
                         # + float(args.Qsel) * (percentiles - 0.5))
  halos.sort(order='mass')
  mock1 = halos[-Ncluster:] #only choose most massive clusters
  dummy2 = h5.File(str(mock_file2), 'r')
  mock2 = dummy2['particles']
