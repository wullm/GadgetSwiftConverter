#Convert Gadget-1 format to SWIFT HDF format
#Willem Elbers, 14/01/2021

import numpy as np
import h5py
import sys, os

# The Gadget-1 file format consists of blocks, preceded and succeeded by a
# 4-byte integer indicating the total number of bytes in the block, i.e.
# SIZE BLOCK SIZE. The first 5 blocks are: Header, Position, Velocity, Id, Mass


#Read every attribute from the header
def read_header(f):
    f.seek(0) #jump to start
    header = {}
    header["header_bytes"] = np.fromfile(f, dtype=np.int32, count=1)[0]
    header["Npart"] = np.fromfile(f, dtype=np.uint32, count=6)
    header["Massarr"] = np.fromfile(f, dtype=np.double, count=6)
    header["Time"] = np.fromfile(f, dtype=np.double, count=1)[0]
    header["Redshift"] = np.fromfile(f, dtype=np.double, count=1)[0]
    header["FlagSfr"] = np.fromfile(f, dtype=np.int32, count=1)[0]
    header["FlagFeedback"] = np.fromfile(f, dtype=np.int32, count=1)[0]
    header["Nall"] = np.fromfile(f, dtype=np.int32, count=6)
    header["FlagCooling"] = np.fromfile(f, dtype=np.int32, count=1)[0]
    header["NumFiles"] = np.fromfile(f, dtype=np.int32, count=1)[0]
    header["BoxSize"] = np.fromfile(f, dtype=np.double, count=1)[0]
    header["Omega0"] = np.fromfile(f, dtype=np.double, count=1)[0]
    header["OmegaLambda"] = np.fromfile(f, dtype=np.double, count=1)[0]
    header["HubbleParam"] = np.fromfile(f, dtype=np.double, count=1)[0]
    header["FlagAge"] = np.fromfile(f, dtype=np.int32, count=1)[0]
    header["FlagMetals"] = np.fromfile(f, dtype=np.int32, count=1)[0]
    header["NallHW"] = np.fromfile(f, dtype=np.int32, count=6)
    header["Flag_Entropy_ICs"] =  np.fromfile(f, dtype=np.int32, count=1)[0]
    return(header)

#Usage message
if (len(sys.argv) < 2 or sys.argv[1] == "--help" or sys.argv[1] == "-h"):
    print("Convert Gadget-1 format to SWIFT HDF format.")
    print("Input filenames are of the format base_filename.x")
    print("Usage: python3 convert.py base_filename")
    os._exit(1)

#The Gadget filenames are of the format "name.x" with x in [0, Nfiles-1]
base_filename = sys.argv[1]

#First, open the first file and read the header
first_file = base_filename + ".0"
f = open(first_file, "rb")
header = read_header(f)
Nfiles = header["NumFiles"]
Ntypes = len(header["Npart"])
f.close()

#Create the output hdf5 format
outname = base_filename + ".hdf5"
g = h5py.File(outname, mode="w")

#Write the header group
g.create_group("Header")
g["Header"].attrs["Npart"] = header["Nall"]
g["Header"].attrs["NumPart_Total"] = header["Nall"]
g["Header"].attrs["NumPart_Total_HighWord"] = header["NallHW"]
g["Header"].attrs["MassTable"] = header["Massarr"]
g["Header"].attrs["BoxSize"] = header["BoxSize"]
g["Header"].attrs["Flag_Entropy_ICs"] = header["Flag_Entropy_ICs"]
g["Header"].attrs["Time"] = header["Time"]
g["Header"].attrs["Redshift"] = header["Redshift"]


#Create the particle data groups
for i in range(Ntypes):
    TotSize = header["Nall"][i]
    if (TotSize > 0):
        GroupName = "PartType" + str(i)
        g.create_group(GroupName)
        g[GroupName].create_dataset("Coordinates", dtype=np.single, shape=(TotSize,3))
        g[GroupName].create_dataset("Velocities", dtype=np.single, shape=(TotSize,3))
        g[GroupName].create_dataset("ParticleIDs", dtype=np.uint64, shape=(TotSize,))
        g[GroupName].create_dataset("Masses", dtype=np.single, shape=(TotSize,))

#A counter for the total number of particles written per type
PartsWritten = np.zeros(Ntypes, dtype=np.uint64)

#Loop over all the files
for k in range(Nfiles):
    fname = base_filename + "." + str(k)
    print("Reading", fname)

    #Open the file and read the header
    f = open(fname, "rb")
    header = read_header(f)
    Npart = header["Npart"]

    #Total number of particles in this file across all types
    N = Npart.astype(np.int32).sum()

    #Skip to the particle positions
    f.seek(header["header_bytes"]+4)
    byte_counts = np.fromfile(f, dtype=np.int32, count=2)
    pos = np.fromfile(f, dtype=np.single, count=N * 3)

    #Continue with the velocities
    byte_counts = np.fromfile(f, dtype=np.int32, count=2)
    vel = np.fromfile(f, dtype=np.single, count=N * 3)

    #Continue with the ids
    byte_counts = np.fromfile(f, dtype=np.int32, count=2)
    ids = np.fromfile(f, dtype=np.uint64, count=N)

    #Continue with the masses
    byte_counts = np.fromfile(f, dtype=np.int32, count=2)
    mass = np.fromfile(f, dtype=np.single, count=N)

    #Close the file
    f.close()

    #Separate out the particle data by type
    pos_per_type = {}
    vel_per_type = {}
    ids_per_type = {}
    mass_per_type = {}

    Ntypes = len(Npart)

    for i in range(Ntypes):
        start = Npart[:i].astype(np.int32).sum()
        end = start + Npart[i]
        pos_per_type[i] = pos[start*3:end*3].reshape((-1,3))
        vel_per_type[i] = vel[start*3:end*3].reshape((-1,3))
        ids_per_type[i] = ids[start:end]
        mass_per_type[i] = mass[start:end]

    #Write the particle data per type
    for i in range(Ntypes):
        Size = Npart[i]
        if (Size > 0):
            start = PartsWritten[i]
            end = PartsWritten[i] + Size
            GroupName = "PartType" + str(i)
            g[GroupName + "/Coordinates"][start:end,] = pos_per_type[i]
            g[GroupName + "/Velocities"][start:end,] = vel_per_type[i]
            g[GroupName + "/ParticleIDs"][start:end,] = ids_per_type[i]
            if (mass_per_type[i].shape[0] > 0):
                g[GroupName + "/Masses"][start:end,] = mass_per_type[i]
            PartsWritten[i] += Size

#Close the file
g.close()
