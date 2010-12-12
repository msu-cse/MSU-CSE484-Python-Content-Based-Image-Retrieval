"""
Does all of the index-building, cluster-generating, and bag-of-words generating steps in one go.
"""
from cbir import *
import sys
import pyflann
import numpy
from os.path import exists,basename
from os import mkdir
from buildIndex import buildIndex

log = logging.getLogger('cbir.bag')

if __name__ == '__main__':
    optionParser = OptionParser(description=__doc__)
    optionParser.add_option('-f',
        dest='feature_file',
        help="Feature file, e.g. 'esp.feature'")
    optionParser.add_option('-c',
        dest='cluster_file',
        help="File containing array of clusters",
        default='')
    optionParser.add_option('-l',
        dest='image_list',
        help="File containing a list of images, one per line")
    optionParser.add_option('-s',
        dest='feature_size_list',
        help="File containing the number of key points per image, one per line, in the same order as image_list")
    optionParser.add_option('-d',
        dest='output_dir',
        help="Directory to write the output files to")        
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
    (opts,args) = optionParser.parse_args()
    
    # -- Usage
    if None in [opts.feature_file,
#                opts.cluster_file,
                opts.image_list,
                opts.feature_size_list, 
                opts.output_dir,
                opts.num_clusters,
                opts.num_iterations]:
        optionParser.print_help()
        sys.exit(0)
        
    # -- Create the output directory
    if not exists(opts.output_dir):
        mkdir(opts.output_dir)

    # -- Read in the list of files
    image_list = file(opts.image_list,'r')
    image_filenames = [line.strip() for line in image_list.readlines()]

    # -- Read in the number of features per file
    feature_count = dict()
    feature_size_list = file(opts.feature_size_list,'r')
    for image in image_filenames:
        feature_count[image] = int( feature_size_list.readline() )
    log.info( "Loaded information for %i images" % len(feature_count) )

    # -- Sanity check.  The first image, has 380 features.
    assert 380 == feature_count['00004fc7eb4bf00ba434c167890b99fa.jpg']

    # -- Load the feature file
    log.info("Loading features from file...")
    featureSet = float32(pyflann.io.dataset.load(filename=opts.feature_file))
    log.debug("Loaded %s rows" % len(featureSet))
    
    # -- Calculate the clusters
    # Parameters taken from the project PDF
    clusters = None
    # clusteringFlann = buildIndex(featureSet)
    clusteringFlann = FLANN(target_precision=1,
            build_weight=0.01,
            memory_weight=1,
            cb_index=0.06,
            checks=2048,
            branching=10,
            log_level='info')
    
    if exists(opts.cluster_file):
        log.info("Loading clusters from file")
        cluster_file = file(opts.cluster_file,'r')
        base = basename(opts.cluster_file)
        clusters = imp.load_source(base,opts.cluster_file).clusters
    else:
        log.info("Calculating %s clusters (%s passes)" % (opts.num_clusters,opts.num_iterations))        
        clusters = clusteringFlann.kmeans(
            pts=featureSet,
            num_clusters=opts.num_clusters,
            dtype=float32,
            max_iterations=opts.num_iterations,
            centers_init='gonzales')

    # -- Load the clusters into FLANN as if they were the points
    log.info("Building cluster index...")
    clusterIndex = buildIndex(clusters)

    # -- For each point in the feature list, find its 'nearest neighbor',
    # i.e. which cluster it belongs to.
    log.info("Calculating distances...")
    cluster_list,distance_list = clusterIndex.nn_index(featureSet,num_neighbors=1)

    log.info( "Have %i nearest-clusters, writing files" % len(cluster_list))

    # -- For each file, write its words to [out_dir]/[image].txt
    current_feature = 0
    for filename in image_filenames:
        base = basename(filename).replace('jpg','txt')
        out = file("%s/%s" % (opts.output_dir,base), 'w+')

        for feature in xrange(feature_count[filename]):

            # -- If we are using an abbreviated index/cluster list, there will
            # not be information for EVERY file.
            if current_feature >= len(cluster_list): break

            # -- Write the word
            out.write( "%s\n" % cluster_list[current_feature] )
            current_feature += 1

        # -- If we didn't write anything, delete the file
        if out.tell() == 0:
            os.remove(out.name)
        else:
            out.close()