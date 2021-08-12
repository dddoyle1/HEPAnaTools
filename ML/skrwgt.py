import numpy as np
import pandas as pd
import os
from pyHEPTools.ML.dataloader import TrainTestDataLoader
from pyHEPTools.ML.skclassifier import *
from scipy.stats import ks_2samp

class SKReweighter(SKClassifier):
    def Weights(self, X):
        return self._wgt_func(super().Eval(X))

    def PlotWeights(self, plots):
        import matplotlib.pyplot as plt
        max_weight = max(self._train_weights.max(), self._test_weights.max())

        _, bins, _ = plt.hist(self._train_weights[self._dataloader._train_Y==1],
                              bins=100,
                              range=(0, max_weight),
                              histtype='stepfilled',
                              alpha=0.7,
                              color='blue',
                              label='S (Train)',
                              density=True)
        plt.hist(self._train_weights[self._dataloader._train_Y==0],
                 bins=bins,
                 edgecolor='r',
                 hatch='/',
                 histtype='step',
                 label='B (Train)',
                 density=True)

        hist, _ = np.histogram(self._test_weights[self._dataloader._test_Y==0],
                               bins=bins,
                               density=True)
        bin_widths = np.diff(bins)
        bin_centers = (bins[:-1] + bins[1:]) / 2

        plt.errorbar(bin_centers,
                     hist,
                     xerr=bin_widths / 2,
                     fmt='.',
                     c='r',
                     label='B (Test)')
                                  
        hist, _ = np.histogram(self._test_weights[self._dataloader._test_Y==1],
                               bins=bins,
                               density=True)
        plt.errorbar(bin_centers, hist, xerr=bin_widths / 2,
                     fmt='.',
                     c='b',
                     label='S (Test)')
        plt.xlabel('%s weights' % self._name)
        plt.ylabel('Events (Area Normalized)')
        plt.legend(loc='best')

        plt.savefig(os.path.join(plots, 'weights.pdf'))
        print('Wrote %s' % os.path.join(plots, 'weights.pdf'))

        plt.close()

    def PlotReweightedInputVariableDistributions(self, plots):
        import matplotlib.pyplot as plt
        for col in self._dataloader._feature_vars + self._dataloader._spectator_vars:
            minx  = self._dataloader._train_X[col].min()
            maxx  = self._dataloader._train_X[col].max()
            nbins = 50

            fig, ax = plt.subplots(figsize=(10, 5), nrows=2, ncols=2, gridspec_kw={'height_ratios': [2,1], 'hspace':0}, sharex=True)

            for icol, sample in enumerate(['training', 'testing']):
                if sample == 'training':
                    nom_array = self._dataloader._train_X[col][self._dataloader._train_Y==0].values
                    sig_array = self._dataloader._train_X[col][self._dataloader._train_Y==1].values
                    weights = self._train_weights [self._dataloader._train_Y ==0]
                elif sample == 'testing':
                    nom_array = self._dataloader._test_X [col][self._dataloader._test_Y ==0].values
                    sig_array = self._dataloader._test_X[col][self._dataloader._test_Y==1].values
                    weights = self._test_weights [self._dataloader._test_Y ==0]
                    
                nnom, bins, _ = ax[0][icol].hist(nom_array,
                                                range=(minx, maxx),
                                                bins=nbins,
                                                color='b',
                                                histtype='step',
                                                label='Background')
                nnom = nnom.astype(float)
                nrwgted, _, _ = ax[0][icol].hist(nom_array,
                                                weights=weights,
                                                bins=bins,
                                                color='r',
                                                histtype='step',
                                                label='Reweighted Background')

                n = np.histogram(sig_array, bins=bins)[0].astype(float)
                bin_widths = np.diff(bins)
                bin_centers = (bins[:-1] + bins[1:]) / 2
                ax[0][icol].errorbar(bin_centers, n, xerr=bin_widths / 2, fmt='.', c='k', label='S')
                leg = ax[0][icol].legend(loc='best')
                ax[0][icol].set_xlim([minx, maxx])
                ax[1][icol].set_xlim([minx, maxx])
                ax[0][icol].set_ylabel('Events')
                ax[1][icol].set_ylabel('Ratio')
                ax[1][icol].set_xlabel(col)
                ax[1][icol].set_ylim([0.5, 1.5])
                ax[0][icol].set_title(sample)

                
                # cheat empty bins                
                n[n==0] = 1e-10
                nrwgted[nrwgted==0] = 1e-10
                nnom[nnom==0] = 1e-10
                    
                rwgt_ratio = nrwgted / n
                nom_ratio  = nnom / n

                nom_chi2 = (np.power(nnom - n, 2) / n).sum()
                rwgt_chi2 = (np.power(nrwgted - n, 2) / n).sum()
            
                leg.get_texts()[0].set_text(r'B             $\chi^2 / dof = %.1f / %d$' % (nom_chi2 , nbins))
                leg.get_texts()[1].set_text(r'B (Rwgt) $\chi^2 / dof = %.1f / %d$' % (rwgt_chi2, nbins))

                nom_ratio_error = np.sqrt(1 / n + 1 / nnom)
                
                def sumw2(n, weights, bins):
                    bin_idxs = np.digitize(n, bins)
                    
                    return np.array([(weights[np.where(bin_idxs==i)]**2).sum() for i in np.arange(1, len(bins))])


                w2 = sumw2(nom_array, weights, bins)
                w2[w2==0] = 1e-10
                try:
                    rwgt_ratio_error = np.sqrt(1 / w2 + 1 / n)
                except FloatingPointError as err:
                    print(w2)
                    print(n)
                    raise err

                ax[1][icol].axhline(1, ls='--', color='gray')
                ax[1][icol].errorbar(bin_centers, nom_ratio,
                                     xerr=bin_widths / 2,
                                     yerr=nom_ratio_error,
                                     fmt='.', ms=3,
                                     c='b')            
                ax[1][icol].errorbar(bin_centers, rwgt_ratio,
                                     xerr=bin_widths / 2,
                                     yerr=rwgt_ratio_error,
                                     fmt='.', ms=3,
                                     c='r')
            var_type = 'feature' if col in self._dataloader._feature_vars else 'spectator'
            plot_path = os.path.join(plots, 'reweighted_%s_var_%s.pdf' % (var_type, col))
            plt.savefig(plot_path)
            print('Wrote %s' % plot_path)
            plt.close()
        
    
    def Train(self, *args, **kwargs):
        training_report, testing_report = super().Train(*args, **kwargs)

        self._train_weights = self._wgt_func(self._prob_train)
        self._test_weights = self._wgt_func(self._prob_test)

        chi2_train = 0
        chi2_test  = 0
        nbins_tot = 0
        for col in self._dataloader._train_X.columns:
            minx  = self._dataloader._train_X[col].min()
            maxx  = self._dataloader._train_X[col].max()
            nbins = 50
            nbins_tot += nbins
            
            n_nom_train , bins = np.histogram(self._dataloader._train_X[col][self._dataloader._train_Y==0],
                                              bins=nbins,
                                              range=(minx, maxx))
            n_syst_train , _ = np.histogram(self._dataloader._train_X[col][self._dataloader._train_Y==1],
                                            bins=bins)

            n_nom_rwgt_train, _ = np.histogram(self._dataloader._train_X[col][self._dataloader._train_Y==0],
                                               bins=bins,
                                               weights=self._train_weights[self._dataloader._train_Y==0])


            n_nom_test , bins = np.histogram(self._dataloader._test_X[col][self._dataloader._test_Y==0],
                                              bins=nbins,
                                              range=(minx, maxx))
            n_syst_test , _ = np.histogram(self._dataloader._test_X[col][self._dataloader._test_Y==1],
                                            bins=bins)

            n_nom_rwgt_test, _ = np.histogram(self._dataloader._test_X[col][self._dataloader._test_Y==0],
                                               bins=bins,
                                               weights=self._test_weights[self._dataloader._test_Y==0])

            # handle infinities
            diff_train = n_nom_rwgt_train - n_syst_train
            diff_test  = n_nom_rwgt_test  - n_syst_test

            diff_train[n_syst_train==0] = 1
            diff_test [n_syst_test ==0] = 1

            n_syst_train[n_syst_train==0] = 1
            n_syst_test [n_syst_test ==0] = 1
            
            chi2_train += (np.power(diff_train, 2) / n_syst_train).sum()
            chi2_test  += (np.power(diff_test , 2) / n_syst_test ).sum()

            chi2_test  += np.nan_to_num(np.power((n_nom_rwgt_test  - n_syst_test ), 2) / n_syst_test ,
                                        nan=1, posinf=1, neginf=1).sum()

        training_report['rwgt_chi2'] = chi2_train
        training_report['rwgt_dof'] = nbins_tot
        testing_report ['rwgt_chi2'] = chi2_test
        testing_report ['rwgt_dof'] = nbins_tot

        return training_report, testing_report

class BDTReweighter(SKReweighter, BDTClassifier):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        def wgt_func(probs):
            return probs / (1 - probs)
        self._wgt_func = wgt_func
        
class NBReweighter(SKReweighter, NBClassifier):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        def wgt_func(probs):
            return (1 + probs) / (1 - probs)
        self._wgt_func = wgt_func
        
class MLPReweighter(SKReweighter, MLPClassifier):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        def wgt_func(probs):
            return probs / (1 - probs)
        self._wgt_func = wgt_func

class GPReweighter(SKReweighter, GPClassifier):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        def wgt_func(probs):
            return probs / (1 - probs)
        self._wgt_func = wgt_func

class KNNReweighter(SKReweighter, KNNClassifier):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        def wgt_func(probs):
            return probs / (1.00001 - probs)
        self._wgt_func = wgt_func        

class RNReweighter(SKReweighter, RNClassifier):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        def wgt_func(probs):
            return probs / (1.00001 - probs)
        self._wgt_func = wgt_func        

_Types = {'bdt'     : BDTReweighter,
          'gaussnb' : NBReweighter ,
          'mlp'     : MLPReweighter,
          'gp'      : GPReweighter,
          'knn'     : KNNReweighter,
          'rn'      : RNReweighter}

def ReweighterType(model_type):
    return _Types[model_type]

class ManyReweighters(ManyClassifiers):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def EvaluateAll(self, sort_key = lambda report: (-1 * report['rwgt_chi2'],
                                                     report['roc_auc'],
                                                     report['Signal']['KS-pvalue'])):
        super().EvaluateAll(sort_key)
        for model in self._models:
            self._models[model].PlotWeights(os.path.join(self._name, model, 'plots'))
        
    def PlotAllReweightedInputVariableDistributions(self):
        for model in self._models:
            self._models[model].PlotReweightedInputVariableDistributions(os.path.join(self._name, model, 'plots'))
            
