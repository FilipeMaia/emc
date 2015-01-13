import parallel
import os

jobs = range(20)

def calculate_table(sampling_n):
    os.system("python calculate_rotation_table.py -n {0} /Volumes/ekeberg/Work/programs/emc/rotations/rotations_{0}.h5 data/rotations_table_n{0}.h5".format(sampling_n))

parallel.run_parallel(jobs, calculate_table, 8)
