import numpy as np
import os

"""
Credit to https://betatim.github.io/posts/sklearn-for-TMVA-users/ 
around which this library is designed 

Author: Derek.Doyle@colostate.edu
"""


class TrainTestDataLoader:
    def __init__(
        self,
        feature_vars=None,
        spectator_vars=None,
        train_X=None,
        train_Y=None,
        test_X=None,
        test_Y=None,
    ):
        """
        feature_vars and spectator_vars can be a column names or a pandana.Var
        if pandana.Vars, they will be used to fill spectra in 
        TrainTestDataLoader.Go()
        """
        self._train_X = train_X
        self._train_Y = train_Y
        self._test_X = test_X
        self._test_Y = test_Y

        self._feature_vars = feature_vars if feature_vars else self._train_X.columns
        self._spectator_vars = spectator_vars if spectator_vars else []
        self._signal_cut = None
        self._bkgd_cut = None

    def AddVariable(self, var):
        """
        var can be a column name or a pandana.Var
        if pandana.Var, it is used to fill spectra in 
        TrainTestDataLoader.Go()
        """
        self._feature_vars.append(var)

    def AddSignalCut(self, cut):
        if self._signal_cut is None:
            self._signal_cut = cut
        else:
            self._signal_cut = self._signal_cut & cut

    def AddBackgroundCut(self, cut):
        if self._bkgd_cut is None:
            self._bkgd_cut = cut
        else:
            self._bkgd_cut = self._bkgd_cut & cut

    def Go(self, loader):
        from pandana import Var, Spectrum

        """
        Build spectra with given feature and signal definitions
        Load data with loader
        Split into testing and training sets
        """

        kFeatures = Var(
            lambda tables: pd.concat([v(tables) for v in self._feature_vars], axis=1)
        )

        signal = Spectrum(loader, self._signal_cut, kFeatures)
        bkgd = Spectrum(loader, self._bkgd_cut, kFeatures)

        loader.Go()

        self._x = pd.concat([signal._df, bkgd._df])
        self._y = pd.Series(index=self._x.index, name="Y")

        self._y.loc[signal._df.index] = 1
        self._y.loc[bkgd._df.index] = 0

    def SaveTo(self, path_or_buf, *args, **kwargs):
        self._x.to_hdf(path_or_buf, "features", *args, **kwargs)
        self._y.to_hdf(path_or_buf, "classification", *args, **kwargs)

    @staticmethod
    def LoadFrom(path_or_buf, *args, **kwargs):
        dataloader = TrainTestDataLoader()
        dataloader._x = pd.read_hdf(path_or_buf, "features", *args, **kwargs)
        dataloader._y = pd.read_hdf(path_or_buf, "classification", *args, **kwargs)

        return dataloader

    def PrepareTrainingAndTestData(
        self, ntrain_signal, ntrain_bkgd, ntest_signal, ntest_bkgd
    ):
        assert self._x is not None and self._y is not None
        from sklearn.model_selection import train_test_split

        self._train_signal, self._test_signal = train_test_split(
            self._x[self._y == 1], test_size=ntest_signal, train_size=ntrain_signal
        )
        self._train_bkgd, self._test_bkgd = train_test_split(
            self._x[self._y == 0], test_size=ntest_signal, train_size=ntrain_signal
        )

        self._train_X = pd.concat([self._train_signal, self._train_bkgd])
        self._train_Y = self._y.loc[self._train_X.index]

        self._test_X = pd.concat([self._test_signal, self._test_bkgd])
        self._test_Y = self._y.loc[self._test_X.index]
