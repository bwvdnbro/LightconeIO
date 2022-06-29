import numpy as np
import h5py
import yaml
from astropy.cosmology import Planck18
import unyt
import os
import re

array_re = re.compile("\Aarray\(dtype=(\w+),shape=\(([0-9,]+)\)\)\Z")

with open("lightcone_particle_file_metadata.yml", "r") as handle:
    pmetadata = yaml.safe_load(handle)


def z_to_Mpc(z):
    return unyt.unyt_array.from_astropy(Planck18.comoving_distance(z)).to("Mpc").value


def decode_attribute_type(value, typestr):
    if typestr == "string":
        return value
    elif "array(" in typestr:
        m = array_re.match(typestr)
        dtype = m.group(1)
        return np.array(value, dtype=getattr(np, dtype))
    else:
        raise AttributeError(f"Unknown type: {typestr}!")


nrank = 4
nfile_per_rank = 5

nshell = 3
zmin = 0.0
zmax = 0.15
zrange = np.linspace(zmin, zmax, nshell + 1)
shell_inner_radii = z_to_Mpc(zrange[:-1])
shell_outer_radii = z_to_Mpc(zrange[1:])

center = np.array([750.0, 750.0, 750.0])

index_attrs = {
    "final_particle_file_on_rank": np.zeros(nrank, dtype=np.int32) + nfile_per_rank - 1,
    "maximum_redshift_BH": np.array([15.0]),
    "maximum_redshift_DM": np.array([0.25]),
    "maximum_redshift_DMBackground": np.array([-1.0]),
    "maximum_redshift_Gas": np.array([0.5]),
    "maximum_redshift_Neutrino": np.array([0.25]),
    "maximum_redshift_Sink": np.array([-1.0]),
    "maximum_redshift_Stars": np.array([0.5]),
    "minimum_redshift_BH": np.array([0.0]),
    "minimum_redshift_DM": np.array([0.0]),
    "minimum_redshift_DMBackground": np.array([0.0]),
    "minimum_redshift_Gas": np.array([0.0]),
    "minimum_redshift_Neutrino": np.array([0.0]),
    "minimum_redshift_Sink": np.array([0.0]),
    "minimum_redshift_Stars": np.array([0.0]),
    "nr_files_per_shell": np.array([1]),
    "nr_mpi_ranks": np.array([nrank], dtype=np.int32),
    "nr_shells": np.array([nshell], dtype=np.int32),
    "observer_position": center,
    "shell_inner_radii": shell_inner_radii,
    "shell_outer_radii": shell_outer_radii,
}
lightcone_attrs = {}
for attr in pmetadata["Lightcone"]["Attributes"]:
    a = pmetadata["Lightcone"]["Attributes"][attr]
    lightcone_attrs[attr] = decode_attribute_type(a["Value"], a["Type"])
    if attr in index_attrs and attr != "nr_mpi_ranks":
        lightcone_attrs[attr] = index_attrs[attr]

lightcone_attrs["nr_mpi_ranks"][0] = nrank

with h5py.File("lightcone0_index.hdf5", "w") as handle:
    group = handle.create_group("Lightcone")
    for key in index_attrs:
        group.attrs[key] = index_attrs[key]

np.random.seed(42)
ptypes = ["BH", "DM", "Gas", "Neutrino", "Stars"]
npart = 100000
pz = zmin + np.random.random(npart) * (zmax - zmin)
pa = 1.0 / (1.0 + pz)
pr = z_to_Mpc(pz)
costheta = 2.0 * np.random.random(npart) - 1.0
sintheta = np.sqrt((1.0 - costheta) * (1.0 + costheta))
phi = 2.0 * np.pi * np.random.random(npart)
cosphi = np.cos(phi)
sinphi = np.sin(phi)
coords = np.zeros((npart, 3))
coords[:, 0] = pr * sintheta * cosphi
coords[:, 1] = pr * sintheta * sinphi
coords[:, 2] = pr * costheta**3
# do not recentre, positions are relative w.r.t. observer position

ptype = np.random.choice(ptypes, size=npart, p=[0.01, 0.4, 0.2, 0.1, 0.29])
prank = np.random.choice(np.arange(nrank, dtype=np.int32), size=npart)

os.makedirs("lightcone0_particles", exist_ok=True)

file_z = np.linspace(zmin, zmax, nfile_per_rank + 1)
file_zmin = file_z[:-1][::-1]
file_zmax = file_z[1:][::-1]
cumulative_counts = [{}] * nrank
for type in ptypes:
    for irank in range(nrank):
        cumulative_counts[irank][type] = 0
for ifile in range(nfile_per_rank):
    for irank in range(nrank):
        filemask = (pz >= file_zmin[ifile]) & (pz < file_zmax[ifile]) & (prank == irank)
        filetypes = ptype[filemask]
        filecoords = coords[filemask]
        filepa = pa[filemask]
        filenpart = {}
        filetypemask = {}
        for type in ptypes:
            filetypemask[type] = filetypes == type
            filenpart[type] = filetypemask[type].sum()
            filename = f"lightcone0_particles/lightcone0_{ifile:04d}.{irank}.hdf5"
        lightcone_attrs["file_index"][0] = ifile
        lightcone_attrs["mpi_rank"][0] = irank
        lightcone_attrs["expansion_factor"][0] = 1.0 / (1.0 + file_zmin[ifile])
        for type in ptypes:
            cumulative_counts[irank][type] += filenpart[type]
            lightcone_attrs[f"cumulative_count_{type}"][0] = cumulative_counts[irank][
                type
            ]
        with h5py.File(filename, "w") as handle:
            for group in pmetadata.keys():
                if group in ptypes and filenpart[group] == 0:
                    continue
                group_handle = handle.create_group(group)
                if group == "Lightcone":
                    for attr in lightcone_attrs:
                        group_handle.attrs[attr] = lightcone_attrs[attr]
                    continue
                for attr in pmetadata[group]["Attributes"]:
                    a = pmetadata[group]["Attributes"][attr]
                    group_handle.attrs[attr] = decode_attribute_type(
                        a["Value"], a["Type"]
                    )
                for dset in pmetadata[group]["Datasets"]:
                    metadset = pmetadata[group]["Datasets"][dset]
                    dtype = getattr(np, metadset["Type"])
                    if dset == "Coordinates":
                        dsetdata = filecoords[filetypemask[group]].astype(dtype)
                        maxshape = (None, None)
                    elif dset == "ExpansionFactors":
                        dsetdata = filepa[filetypemask[group]].astype(dtype)
                        maxshape = (None,)
                    else:
                        numpart = filenpart[group]
                        dmin = metadset["Min"]
                        dmax = metadset["Max"]
                        if isinstance(dmin, list):
                            dsetdata = np.random.random((numpart, len(dmin)))
                            for i in range(len(dmin)):
                                dsetdata[:, i] *= dmax[i] - dmin[i]
                                dsetdata[:, i] += dmin[i]
                            maxshape = (None, None)
                        else:
                            dsetdata = dmin + np.random.random(numpart) * (dmax - dmin)
                            maxshape = (None,)
                        dsetdata = dsetdata.astype(dtype)
                    dset_handle = group_handle.create_dataset(
                        dset, data=dsetdata, dtype=dtype, maxshape=maxshape
                    )
                    for attr in metadset["Attributes"]:
                        a = metadset["Attributes"][attr]
                        try:
                            dset_handle.attrs[attr] = decode_attribute_type(
                                a["Value"], a["Type"]
                            )
                        except KeyError:
                            dset_handle.attrs[attr] = decode_attribute_type(
                                a["value"], a["Type"]
                            )
