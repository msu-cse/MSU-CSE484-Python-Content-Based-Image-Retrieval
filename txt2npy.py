
from cbir import *
import sys
import numpy
import pyflann
log = logging.getLogger('cbir.txt2hdf5')


if __name__ == '__main__':
    optionParser = OptionParser(description=__doc__)
    optionParser.add_option('-f',
        dest='feature_file',
        help="Feature file, e.g. 'esp.feature'")
    (options,args) = optionParser.parse_args()
    
    if None in [options.feature_file]:
        optionParser.print_help()
        sys.exit(0)
    
    # -- Load from text
    # Note that we have to use uint16 because uint8 throws an error.
    log.info("Loading dataset %s" % options.feature_file)
    data = pyflann.io.dataset.load(filename=options.feature_file,
        dtype=numpy.uint8)
    
    outfile = "%s.npy" % options.feature_file
    log.info("Saving dataset %s" % outfile)
    pyflann.io.dataset.save(filename=outfile,
        dataset=data)