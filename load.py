from pyflann import *
from numpy import *
from numpy.random import *
from optparse import OptionParser
from progressBar import progressBar
import sys

optionParser = OptionParser(usage="%prog [file.index]")
(options,args) = optionParser.parse_args()

if len(args) < 1:
    optionParser.print_usage()
    sys.exit(0)

flann = FLANN()
flann.load_index(args[0])
