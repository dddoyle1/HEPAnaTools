from PandAnaTools.ML import TrainTestDataLoader, BDTClassifier
import argparse

parser = argparse.ArgumentParser('BDT Training example')
parser.add_argument('--dataloader'  , help='Output of a TrainTestDataLoader',
                    default=None)
parser.add_argument('--results'     , help='.model.h5 file output by the BDT object',
                    default=None)
parser.add_argument('--plots', help='Directory in which to place plots',
                    default='./')

args = parser.parse_args()

if args.dataloader:
    dataloader = TrainTestDataLoader.LoadFrom(args.dataloader)
    dataloader.PrepareTrainingAndTestData(1000,1000,1000,1000)

    bdt = BDTClassifier('skclassifier_training_example',
                        max_depth      = 3,
                        ada_boost_beta = 0.5,
                        ntrees         = 800,
                        min_node_size  = 0.05)

    bdt.AddDataLoader(dataloader)
    bdt.Train()
    
elif args.results:
    bdt = BDT.LoadFrom(args.results, 'bdt_training_example')
    bdt.PlotROC                            (args.plots)
    bdt.PlotCompareTrainTest               (args.plots)
    bdt.PlotInputVariableDistributions     (args.plots)
    bdt.PlotInputVariableLinearCorrelations(args.plots)
    
