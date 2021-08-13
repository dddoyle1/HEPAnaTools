from hepanatools.fit.template_fit import *
from hepanatools.fit.utils import mv_covariance, fake_mv
# precompile mv_covariance with a small matrix
mv_covariance(fake_mv(2, 10))

