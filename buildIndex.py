from process import *

if __name__ == '__main__':
    optionParser = OptionParser(usage="%prog [file]")
    (options,args) = optionParser.parse_args()
    f = FeatureList(args[0])
    f.process()
    
    flann = FLANN()
    params = flann.build_index(f.dataset,  
        algorithm='kdtree',
        trees=8,
        target_precision=-1,
        build_weight=0.01,
        memory_weight=1,
        random_seed=-1,
        log_level = "none")

    flann.save_index(f.filename + ".index")
    print "Saved index to %s.index" % f.filename