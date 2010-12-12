"""
Parses a feature file into an in-memory array suitable for processing by FLANN
"""

from cbir import *
import numpy
import pyflann
log = logging.getLogger('cbir.processFeatures')

class FeatureList:
    
    dataset = None
    
    def __init__(self, filename):
        self.filename = filename
    
    def process(self):
        '''Process the features file'''
        
        self.dataset = pyflann.io.dataset.load(filename=self.filename)
        return
        
    def __repr__(self):
        print "FeatureList('%s')" % self.filename


if __name__ == '__main__':
    optionParser = OptionParser(description=__doc__)
    optionParser.add_option('-f',
        dest='feature_file',
        help="Feature file, e.g. 'esp.feature'")
    (options,args) = optionParser.parse_args()
    
    if options.feature_file:
        f = FeatureList(options.feature_file)
    else:
        optionParser.print_help()
        sys.exit(0)

#   _fields_ = [
#       ('algorithm', c_int),
#       ('checks', c_int),
#       ('cb_index', c_float),
#       ('trees', c_int),
#       ('branching', c_int),
#       ('iterations', c_int),
#       ('centers_init', c_int),
#       ('target_precision', c_float),
#       ('build_weight', c_float),
#       ('memory_weight', c_float),
#       ('sample_fraction', c_float),
#       ('log_level', c_int),
#       ('random_seed', c_long),
#   ]
#   _defaults_ = {
#       'algorithm' : 'kdtree',
#       'checks' : 32,
#       'cb_index' : 0.5,
#       'trees' : 1,
#       'branching' : 32,
#       'iterations' : 5,
#       'centers_init' : 'random',
#       'target_precision' : -1,
#       'build_weight' : 0.01,
#       'memory_weight' : 0.0,
#       'sample_fraction' : 0.1,
#       'log_level' : "warning",
#       'random_seed' : -1
# }
#   _translation_ = {
#           "algorithm"     : {"linear"    : 0, "kdtree"    : 1, "kmeans"    : 2, "composite" : 3, "saved": 254, "autotuned" : 255, "default"   : 1},
#       "centers_init"  : {"random"    : 0, "gonzales"  : 1, "kmeanspp"  : 2, "default"   : 0},
#       "log_level"     : {"none"      : 0, "fatal"     : 1, "error"     : 2, "warning"   : 3, "info"      : 4, "default"   : 2}
#   }
