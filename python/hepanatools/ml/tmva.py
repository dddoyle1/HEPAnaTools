import ROOT
import array
import numpy as np

class TMVABDT:
    def __init__(self):
        self.reader = ROOT.TMVA.Reader()

        self.Eval = np.vectorize(
            self._eval,
            otypes=[float],
        )

        
class NueID(TMVABDT):
    def __init__(self, bdt_file_name):
        super().__init__('NueID', bdt_file_name)
        self.shwwidth   = array.array('f', [0])
        self.epi0llt    = array.array('f', [0])
        self.electronid = array.array('f', [0])
        self.shwgap     = array.array('f', [0])

        self.reader.AddVariable('shwwidth'  , self.dedxll)
        self.reader.AddVariable('epi0llt'   , self.scatll)
        self.reader.AddVariable('electronid', self.dedx10cm)
        self.reader.AddVariable('shwgap'    , self.dedx40cm)

        self.Eval = np.vectorize(
            self._eval,
            otypes=[float],
        )

    def _eval(self, 
              shwwidth, 
              epi0llt,
              electronid,
              shwgap):
        self.shwwidth = shwwidth
        self.epi0llt = epi0llt
        self.electronid = electronid
        self.shwgap = shwgap

        self.reader.EvaluateMVA('NueID')

class MuonID(TMVABDT):
    def __init__(self, bdt_file_name):
        super().__init__()
        self.dedxll   = array.array('f', [0])
        self.scatll   = array.array('f', [0])
        self.dedx10cm = array.array('f', [0])
        self.dedx40cm = array.array('f', [0])

        self.reader.AddVariable('DedxLL', self.dedxll)
        self.reader.AddVariable('ScatLL', self.scatll)
        self.reader.AddVariable('Avededxlast10cm', self.dedx10cm)
        self.reader.AddVariable('Avededxlast40cm', self.dedx40cm)

        self.reader.BookMVA('BDTG', bdt_file_name)
    
    def _eval(self, 
              dedxll, 
              scatll,
              dedx10cm,
              dedx40cm):
        self.dedxll = dedxll
        self.scatll = scatll
        self.dedx10cm = dedx10cm
        self.dedx40cm = dedx40cm

        self.reader.EvaluateMVA('BDTG')
