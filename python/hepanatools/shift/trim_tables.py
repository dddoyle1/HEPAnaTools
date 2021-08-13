from hepanatools.shift import EventMatchedTable
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('input')
parser.add_argument('output')
parser.add_argument('--keep_vars', nargs='+', default=None)
parser.add_argument('--keep_cuts', nargs='+', default=None)
parser.add_argument('--keep_samples', nargs='+', default=None)
parser.add_argument('--inspect', nargs='*')

args = parser.parse_args()

if type(args.inspect) is list:
    EventMatchedTable.Inspect(args.input,
                              only=args.inspect)
    
else:
    tables = EventMatchedTable.FromH5(args.input)
    if not args.keep_vars: args.keep_vars = tables.var_labels
    if not args.keep_cuts: args.keep_cuts = tables.cut_labels
    if not args.keep_samples: args.keep_samples = tables.samples
    
    new_tables = tables.Trim(keep_vars=args.keep_vars,
                             keep_cuts=args.keep_cuts,
                             keep_samples=args.keep_samples)

    new_tables.ToH5(args.output)

