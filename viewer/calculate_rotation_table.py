"""Program used to precalculate the tables for projecting responsabilities
to bins on a sphere. The result is used by the rotations_module in the viewer."""
import numpy
from optparse import OptionParser
import rotations
import icosahedral_sphere
import h5py
import sys
import parallel

def closest_coordinate(coordinate, points):
    """Calculate the point in points closest to the given coordinate."""
    return (((points - coordinate)**2).sum(axis=1)).argmax()

def read_rotations(filename):
    """Read a rotations file and return the rotations as quaternions and their
    respective weights."""
    with h5py.File(filename, "r") as file_handle:
        quaternions = file_handle['rotations'][:, :4]
        weights = file_handle['rotations'][:, 4]
    return quaternions, weights

def calculate_table(n_sampling, quaternions, weights=None):
    """Calculate the table."""
    number_of_rotations = quaternions.shape[0]
    list_of_points = icosahedral_sphere.sphere_sampling(n_sampling)
    number_of_bins = len(list_of_points)

    rotated_coordinates = numpy.array([rotations.rotate(quaternion, [1., 0., 0.])
                                       for quaternion in quaternions])
    table = numpy.zeros(number_of_rotations, dtype="float64")
    table_weights = numpy.zeros(number_of_bins, dtype="float64")
    for i, coordinate in enumerate(rotated_coordinates):
        progress = float(i) / float(number_of_rotations)
        sys.stdout.write("\r [{0}{1}] {2:.1f}%".format("#"*int(progress*40.),"-"*(40-int(progress*40.)),
                                                       progress*100.))
        sys.stdout.flush()
        index = closest_coordinate(coordinate, list_of_points)
        table[i] = index
        if not weights is None:
            table_weights[index] += weights[i]
        else:
            table_weights[index] += 1.
    sys.stdout.write("\n")

    return table, table_weights

def save_table(filename, table, weights):
    """Write a hdf5 file containing a rotations table and the weights for each bin."""
    with h5py.File(filename, "w") as file_handle:
        file_handle.create_dataset("table", data=table)
        file_handle.create_dataset("weights", data=weights)

def main():
    """The program."""
    parser = OptionParser(usage="%prog [-n SAMPLING] ROTATIONS_FILE OUTPUT_FILE")
    parser.add_option("-n", action="store", dest="n", type="int", default=None,
                      help="Density of sphere sampling if different from the rotation sampling.")
    parser.add_option("-w", action="store_false", dest="use_weights", help="Use weights.")
    options, args = parser.parse_args()

    rotations_file = args[0]
    output_file = args[1]

    quaternions, weights = read_rotations(rotations_file)

    if options.n is None:
        number_of_rotations = len(quaternions)
        n_sampling = rotations.rots_to_n(number_of_rotations)
    else:
        n_sampling = options.n

    if not options.use_weights:
        weights = None

    table, table_weights = calculate_table(n_sampling, quaternions, weights)

    save_table(output_file, table, table_weights)

if __name__ == "__main__":
    main()
