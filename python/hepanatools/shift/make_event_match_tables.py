import sys
import ROOT
import argparse
import pandas as pd
import h5py
import numpy as np
import collections
from EventMatchedTables import *

parser = argparse.ArgumentParser()
parser.add_argument('input', metavar='event_tree_file.root', help='File containing EventTree')
parser.add_argument('--output', help='Name of file to be created', default='event_matched_tables.h5')
parser.add_argument('--indices', help='Comma separated list of index names used to index rows in the tables and perform event matching.', default='run,subrun,batch,cycle,event,slice')
args = parser.parse_args()    

tables = EventMatchedTable.FromROOT(args.input, index_cols=args.indices.split(','))
tables.ToH5(args.output)
