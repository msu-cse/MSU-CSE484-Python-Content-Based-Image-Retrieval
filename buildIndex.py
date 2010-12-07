"""
Builds the FLANN index from a features file, and saves it to file.
"""

from cbir import *
from processFeatures import FeatureList

log = logging.getLogger('cbir.buildIndex')

def buildIndex(dataset):
    '''
    Builds a FLANN index from the provided FeatureList.
    
    Returns the FLANN object.
    '''
    # -- Process the file
    flann = FLANN()
    params = flann.build_index(dataset,  
        algorithm='kdtree',
        trees=8,
        target_precision=-1,
        build_weight=0.01,
        memory_weight=1,
        random_seed=-1,
        log_level = "info")
        
    log.debug( "Built FLANN index with parameters: %s" % params)
    
    return flann
    

if __name__ == '__main__':
    optionParser = OptionParser(description=__doc__)
    optionParser.add_option('-f',
        dest='feature_file',
        help="Feature file, e.g. 'esp.feature'")
    optionParser.add_option('-o',
        dest="output_file",
        help="Index output file (default is <feature>.index)")
    (options,args) = optionParser.parse_args()
    
    # -- Usage
    if options.feature_file is None:
        optionParser.print_help()
        sys.exit(0)
    
    # -- Load the file into memory
    log.info("Loading the feature file...")
    f = FeatureList(options.feature_file)
    f.process()
    log.debug("Loaded %s rows" % len(f.dataset))
    
    # -- Process the file
    log.info("Building index...")
    flann = buildIndex(f.dataset)

    # -- Save the index to file
    if options.output_file is None:
        options.output_file = "%s.index" % options.feature_file
    log.info( "Saving index to %s" % options.output_file )
    flann.save_index(options.output_file)