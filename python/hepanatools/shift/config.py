import configargparse
import numpy as np
import hepanatools
import os

def parse_cdf_config(args=None):
    parser = configargparse.ArgParser('Configuration file for CDF optimization')
    parser.add_argument('-c', is_config_file=True)
    parser.add_argument('--name'        , required=True , help='Name this cdf')
    parser.add_argument('--event_tables', required=True, help='Path to h5 file containing event tables')
    parser.add_argument('--var_label' , required=True , help='Name of var as it appears in event tables')
    parser.add_argument('--cut_label' , required=True , help='Name of cut as it appears in event tables')
    parser.add_argument('--nominal_sample', required=False, help='Name of the nominal sample', default='nominal')
    parser.add_argument('--shifted_sample', required=True , help='Name of systematic sample as it appears in event tables')
    parser.add_argument('--xbins'     , required=True , help='Number of bins along x-axis to optimize', type=int)
    parser.add_argument('--ybins'     , required=True , help='Number of bins along the y-axis (pdf axis)', type=int)
    parser.add_argument('--ybins_func', required=False, help='Name of function to determine y-axis (pdf axis) binning',
                        default='hepanatools.shift.pdf.ybins1')
    parser.add_argument('--bounds'    , required=False, nargs=2, help='Bounds on shifted var. Use \'ninf\'/\'pinf\' for one-sided bounds')
    parser.add_argument('--xlim'      , required=True , nargs=2, help='Limits on x-axis (conditional axis)')
    parser.add_argument('--minimizer_opts', required=False, help='Dict-style string containing additional options to pass to the minimizer')
    parser.add_argument('--objective_bins', help='Bins edges over which to calculate objective function. Exclusive with --objective_bins_eval')
    parser.add_argument('--objective_bins_eval', help='Executable string to determine objective bins, eg. np.linspace(...). Exclusive with --objective_bins')

    parser.add_argument('--train_size', help='If in the interval (0,1], that fraction of events are used for training. If larger than one, that number of events are used for training.', type=float)
    parser.add_argument('--train_split_seed', help='Seed passed to train_test_split for reproducible training samples', default=None, type=int)

    parser.add_argument('--plots', required=True, help='A place to put plots')
    
    if args:
        # parse "args"
        config = parser.parse_args(args)
    else:
        # parse sys.argv
        config = parser.parse_args()

    if config.ybins_func: config.ybins_func = eval(config.ybins_func)
    if config.minimizer_opts: config.minimizer_opts = eval(config.minimizer_opts)
    if config.objective_bins and config.objective_bins_eval:
        raise RuntimeError('--objective_bins and --objective_bins_eval are mutually exclusive')
    if config.objective_bins_eval:
        config.objective_bins = eval(config.objective_bins_eval)

    config.xlim = np.array(config.xlim, dtype=float)
    if config.bounds:
        config.bounds = np.array(config.bounds)
        config.bounds = np.where(config.bounds == 'pinf',  np.inf, config.bounds)
        config.bounds = np.where(config.bounds == 'ninf', -np.inf, config.bounds)
        config.bounds = config.bounds.astype(float)
        config.bounds = hepanatools.shift.pdf.Bounds(*config.bounds)
    else:
        config.bounds = hepanatools.shift.pdf.PassThrough()
        
    if not os.path.isdir(config.plots): os.makedirs(config.plots)
        
    parser.print_values()
    return config