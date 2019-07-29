import Nio
fname = "W_NAFP_C_ECMF_20190114230305_P_C7H011700000204____1.grib"
f = Nio.open_file(fname)
print(f.variables)
