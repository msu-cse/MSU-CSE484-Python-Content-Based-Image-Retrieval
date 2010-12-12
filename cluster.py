'''
Finds the clusters of a given index, and saves those clusters to a file.

The clusters are saved as a 'numpy' array, and the output file is directly
executable as Python source.
'''
from cbir import *
from numpy import set_printoptions,nan
from processFeatures import FeatureList


log = logging.getLogger('cbir.findNeighbors')

if __name__ == '__main__':
    optionParser = OptionParser(description=__doc__)
    optionParser.add_option('-f',
        dest='feature_file',
        help="Feature file, e.g. 'esp.feature'")
    optionParser.add_option('-i',
        dest='index_file',
        help="Index file, e.g. 'esp.feature.index'")
    optionParser.add_option('-o',
        dest='output_file',
        help="Cluster output file, e.g. 'esp.feature.clusters'")
    optionParser.add_option('-n',
        dest='num_clusters',
        help="Number of clusters",
        type=int,
        default=150000)
    optionParser.add_option('-t',
        dest='num_iterations',
        help="Number of k-means iterations",
        type=int,
        default=15)
    (options,args) = optionParser.parse_args()
    
    # -- Usage
    if None in [options.feature_file, options.index_file, options.num_clusters]:
        optionParser.print_help()
        sys.exit(0)
        
    # -- Load the feature file
    log.info("Loading the feature file...")
    f = FeatureList(options.feature_file)
    f.process()
    log.debug("Loaded %s rows" % len(f.dataset))
    
    # -- Load the index
    log.info("Loading the index file...")
    flann = FLANN(log_level='info')
    flann.load_index(options.index_file,f.dataset)
    
    # -- Cluster all of the points
    log.info("Clustering into %i clusters, over %i iterations" %
        (options.num_clusters,options.num_iterations))
    cluster_data = flann.kmeans(f.dataset,options.num_clusters,dtype=uint8,
        max_iterations=options.num_iterations)
    
    # -- Save the clusters to file
    set_printoptions(threshold=nan)
    if options.output_file is None:
        options.output_file = "%s.%i.clusters" % (options.feature_file, options.num_clusters)
    log.info("Writing clusters to %s" % options.output_file)
    f = file(options.output_file,'w+')
    f.write('from numpy import array,uint8\n')
    f.write("clusters = ")
    f.write("%r" % cluster_data)
    f.close()