from HEPAnaTools.Fit.template_fit import *
from HEPAnaTools.Fit.utils import mv_covariance, fake_mv
# precompile mv_covariance with a small matrix
mv_covariance(fake_mv(2, 10))

