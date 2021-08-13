import os
import pandas as pd
import collections
import h5py
import gc

KL = ['run','subrun','batch','cycle','evt','slc']

class EventMatchedTable:
    def __init__(self, tables):
        self.tables = tables
        self.var_labels = list(self.tables.keys())
        self.cut_labels = list(self.tables[self.var_labels[0]].keys())
        self.samples    = list(self.tables[self.var_labels[0]][self.cut_labels[0]].columns)
        self.index_cols = list(self.tables[self.var_labels[0]][self.cut_labels[0]].index.names)
        
    def ToH5(self, output_file):
        if type(output_file) is not h5py.File: output_file = h5py.File(output_file, 'w')
        header = output_file.get('/').attrs
        header['var_labels'] = self.var_labels
        header['cut_labels'] = self.cut_labels
        header['samples'] = self.samples
        header['index_cols'] = self.index_cols

        for var_label in self.var_labels:
            var_group = output_file.create_group(var_label)                
            for cut_label in self.cut_labels:
                cut_group = var_group.create_group(cut_label)

                self.tables[var_label][cut_label] = self.tables[var_label][cut_label].reset_index()
                
                for col in self.tables[var_label][cut_label].columns:
                    cut_group.create_dataset(col,
                                             data=self.tables[var_label][cut_label][col].values,
                                             compression='gzip')
                output_file.flush()
                self.tables[var_label][cut_label] = self.tables[var_label][cut_label].set_index(self.index_cols)
        output_file.close()
        
    def __getitem__(self, key):
        return self.tables[key]
    def keys(self):
        return self.tables.keys()

    @staticmethod
    def Inspect(input_file, only=[]):
        if only and type(only) is not list: only = list(only)        
        if type(input_file) is not h5py.File: input_file = h5py.File(input_file, 'r')
        ascii_ex = {'|': '\u2502',
                    '|--': '\u251c\u2500\u2500 ',
                    '+--': '\u2514\u2500\u2500 '}
        pad = '   '
        header = input_file.get('/').attrs

        if len(only) > 0:
            for prop in only:
                props = header[prop].astype(str).tolist()
                print(prop)
                for p in props:
                    print('%s%s' % (ascii_ex['|--'] if p is not props[-1] else ascii_ex['+--'],
                                    p))
                    

        else:
            var_labels = header['var_labels'].astype(str).tolist()
            cut_labels = header['cut_labels'].astype(str).tolist()
            samples = header['samples'].astype(str).tolist()
            index_cols = header['index_cols'].astype(str).tolist()
        
            for var_label in var_labels:
                is_last_var = var_label is var_labels[-1]
                print(var_label)
                for cut_label in cut_labels:
                    is_last_cut = cut_label is cut_labels[-1]
                    print('%s%s' % (ascii_ex['+--'] if is_last_cut else ascii_ex['|--'],
                                    cut_label))                
                    for sample in samples:

                        print('%s%s%s%s' % (pad if is_last_cut else ascii_ex['|'],
                                            pad,
                                            ascii_ex['+--'] if sample is samples[-1] else ascii_ex['|--'],
                                            sample))
        input_file.close()
                    
    
    @staticmethod
    def FromH5(input_file):
        if type(input_file) is not h5py.File: input_file = h5py.File(input_file, 'r')
        header = input_file.get('/').attrs
        var_labels = header['var_labels'].astype(str).tolist()
        cut_labels = header['cut_labels'].astype(str).tolist()
        samples = header['samples'].astype(str).tolist()
        index_cols = header['index_cols'].astype(str).tolist()

        tables = collections.defaultdict(dict)
        for var_label in var_labels:
            for cut_label in cut_labels:
                group = input_file.get(var_label).get(cut_label)
                data = {col: group.get(col)[()] for col in group.keys()}
                tables[var_label][cut_label] = pd.DataFrame(data)
                tables[var_label][cut_label].set_index(index_cols, inplace=True)
        
        input_file.close()
        return EventMatchedTable(tables)

    def Trim(self,
             keep_vars,
             keep_cuts,
             keep_samples):
        tables = collections.defaultdict(dict)
        for var in keep_vars:
            for cut in keep_cuts:
                tables[var][cut] = pd.concat([self.tables[var][cut][sample] for sample in keep_samples],
                                             axis=1)
        return EventMatchedTable(tables)

    def ApplyCut(self, cut_func):
        cut_df = cut_func(self)

        for var_label in self.var_labels:
            for cut_label in self.cut_labels:
                self.tables[var_label][cut_label] = self.tables[var_label][cut_label].loc[cut_df.to_numpy()]    


    @staticmethod
    def FromROOT(root_file, index_cols):
        import ROOT
        # read file
        if type(root_file) is not ROOT.TFile: root_file = ROOT.TFile.Open(root_file)

        # top level TDirectories contain TTrees for each sample
        samples = [key.GetName() for key in root_file.GetListOfKeys() if key.GetClassName() == 'TDirectoryFile']
        
        # check compatability of all of the trees
        # check all have a tree for each of the same cuts
        cut_labels = None
        for sample in samples:
            _cut_labels = [key.GetName() for key in root_file.GetDirectory(sample).GetListOfKeys() if key.GetClassName() == 'TTree']
            if not cut_labels: 
                cut_labels = _cut_labels
            else: 
                if cut_labels != _cut_labels:
                    print('Error: Incompatible TTrees!')
                    exit(1)

        # check all have a branch for each of the same vars
        var_labels = None
        empty_trees = set()
        for sample in samples:
            for cut_label in cut_labels:
                tree = root_file.Get(os.path.join(sample, cut_label))
                if tree.GetEntries() == 0: empty_trees.add(cut_label)
                _var_labels = [branch.GetName() for branch in tree.GetListOfBranches()]
                if not var_labels:
                    var_labels = _var_labels
                else:
                    if var_labels != _var_labels:
                        print('Error: Incompatible TTrees!')
                        exit(1)

                del tree
                gc.collect()
        var_labels = list(set(var_labels) - set(index_cols))
        cut_labels = list(set(cut_labels) - empty_trees)

        # read validated trees into a dataframe
        tmp_table = collections.defaultdict(dict)
        for sample in samples:
            for cut_label in cut_labels:
                tree = root_file.Get(os.path.join(sample, cut_label))
                tmp_table[sample][cut_label] = EventMatchedTable.tree_to_table(tree,
                                                                               sample, 
                                                                               index_cols)
                del tree
                gc.collect()
        root_file.Close()

        # do the event matching and build table
        tables = collections.defaultdict(dict)
        for cut_label in cut_labels:
            # assume each cut level within each var is the same
            # pre-calculate the intersection for each cut only once
            matched_indices = tmp_table[samples[0]][cut_label][var_labels[0]].index
            for sample in samples[1:]:
                matched_indices = matched_indices.intersection(tmp_table[sample][cut_label][var_labels[0]].index)
            
            for var_label in var_labels:
                before = tmp_table[sample][cut_label][var_label].shape[0]
                after = matched_indices.shape[0]
                for sample in samples:
                    tmp_table[sample][cut_label][var_label] = tmp_table[sample][cut_label][var_label].loc[matched_indices]
            
        for var_label in var_labels:
            for cut_label in cut_labels:
                tables[var_label][cut_label] = pd.concat([tmp_table[sample][cut_label][var_label] for sample in samples],
                                                         axis=1)
                for sample in samples: tmp_table[sample][cut_label][var_label] = pd.Series()

                gc.collect()
                

        return EventMatchedTable(tables)
                        
    # not to be confused with farm-to-table
    def tree_to_table(tree, label, index):
        mat, labels = tree.AsMatrix(return_labels=True)
    
        var_labels = list(set(labels) - set(index))
        index_mat = mat[:, [labels.index(idx) for idx in index if idx in labels] ]

        var_pos = {var: labels.index(var) for var in var_labels if var in labels}
    
        var_mat = mat[:, [var_pos[i] for i in var_pos] ]
        multi_index = pd.MultiIndex.from_tuples(map(tuple, index_mat), names=index)
        series = {var: pd.Series(mat[:, var_pos[var]], index=multi_index, name=label, copy=False) for var in var_pos}
        return series

