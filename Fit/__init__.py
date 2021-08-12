from pyHEPTools.Fit.template_fit import *
from pyHEPTools.Fit.utils import mv_covariance, fake_mv
# precompile mv_covariance with a small matrix
mv_covariance(fake_mv(2, 10))

