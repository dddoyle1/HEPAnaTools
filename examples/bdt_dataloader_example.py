from PandAnaTools.BDT import TrainTestDataLoader
from pandana import *
import argparse

parser = argparse.ArgumentParser('TrainTestDataLoader example')
parser.add_argument('--files', nargs='+', default=None)
parser.add_argument('--limit' , default=None)
parser.add_argument('--stride', default=None)
parser.add_argument('--offset', default=None)
parser.add_argument('--load'  , default=None)


args = parser.parse_args()

assert args.files is None or args.load is None
assert args.files or args.load

if args.files is not None:
    kShwWidth   = Var(lambda tables: tables['rec.vtx.elastic.fuzzyk.png.shwlid'    ]['width'     ])
    kShwGap     = Var(lambda tables: tables['rec.vtx.elastic.fuzzyk.png.shwlid'    ]['gap'       ])
    kElectronID = Var(lambda tables: tables['rec.vtx.elastic.fuzzyk.png.cvnpart'   ]['electronid'])
    kEPi0LLT    = Var(lambda tables: tables['rec.vtx.elastic.fuzzyk.png.shwlid.lid']['epi0llt'   ])

    kNuPDG = Var(lambda tables: tables['rec.mc.nu']['pdg' ])
    kIsCC  = Var(lambda tables: tables['rec.mc.nu']['iscc'])
    kNueCC = (kNuPDG == 12) & (kIsCC == 1)
    kNC    = kIsCC == 0

    dataloader = TrainTestDataLoader()


    dataloader.AddVariable(kShwWidth  )
    dataloader.AddVariable(kShwGap    )
    dataloader.AddVariable(kElectronID)
    dataloader.AddVariable(kEPi0LLT   )

    dataloader.AddSignalCut    (kNueCC)
    dataloader.AddBackgroundCut(kNC   )


    indices = ['run', 'subrun', 'cycle', 'evt', 'subevt', 'rec.vtx.elastic.fuzzyk.png_idx']
    loader = Loader(args.files, 'evt.seq', 'spill', indices=indices)

    dataloader.Go(loader)

    dataloader.SaveTo('bdt_dataloader_example.h5')

elif args.load:
    dataloader = TrainTestDataLoader.LoadFrom(args.load)
    dataloader.PrepareTrainingAndTestData(100,100,100,100)
    

