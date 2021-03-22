"""
Input root file contains genie and ppfx template spectra
Naming scheme: mc_{geniemv,ppfxmv}_template_{i}_{sel,sig}
"""

import sys
import os
import ROOT
import pandas as pd
import numpy as np
import h5py


def write_mv(group, mv):
    for i, imv in enumerate(mv):
        group.create_dataset(str(i), data=imv, compression='gzip')

def hist_to_array(f, hist):
    h = f.Get(hist)
    
    try:
        n = h.GetNbinsX()
    except AttributeError:
        print('Problem loading %s' % hist)
        exit(1)
    return np.array([h.GetBinContent(i) for i in range(1, n+1)])

f = ROOT.TFile.Open(sys.argv[1])

geniemv_sig = [hist_to_array(f, 'mc_geniemv_template_%d_sig/hist' % i) for i in range(300)]
geniemv_sel = [hist_to_array(f, 'mc_geniemv_template_%d_sel/hist' % i) for i in range(300)]
ppfxmv_sel = [hist_to_array(f, 'mc_ppfxmv_template_%d_sel/hist' % i) for i in range(100)]

pot = f.Get('mc_geniemv_template_0_sig/pot').GetBinContent(1)

f.Close()



with h5py.File(os.path.basename(sys.argv[1]).replace('.root', '.h5'), 'w') as out:
    group_geniemv_sig  = out.create_group('geniemv_sig')
    group_geniemv_sel  = out.create_group('geniemv_sel')
    group_ppfxmv_sel   = out.create_group('ppfxmv_sel' )


    group_geniemv_sig.attrs['pot'] = pot
    group_geniemv_sel.attrs['pot'] = pot
    group_ppfxmv_sel .attrs['pot'] = pot


    write_mv(group_geniemv_sig, geniemv_sig)
    write_mv(group_geniemv_sel, geniemv_sel)
    write_mv(group_ppfxmv_sel , ppfxmv_sel )
    
    

    

