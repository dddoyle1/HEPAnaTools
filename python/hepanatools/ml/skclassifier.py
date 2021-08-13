import numpy as np
import pandas as pd
import os
from hepanatools.ml.dataloader import TrainTestDataLoader
from scipy.stats import ks_2samp

class SKClassifier:
    def __init__(self, name, model=None):
        self._name = name
        self._model = model
        
    def AddDataLoader(self, dataloader):
        self._dataloader = dataloader

    @staticmethod
    def Reader(file_name, model):
        import joblib
        model._model = joblib.load(file_name)
        return model

    @staticmethod
    def LoadFrom(file_name, classifier):
        with pd.HDFStore(file_name) as f:
            train = pd.read_hdf(f, 'train')
            test  = pd.read_hdf(f, 'test' )
            classifier._dataloader = TrainTestDataLoader(train_X = train.drop([classifier._name, 'Y'], axis=1),
                                                         train_Y = train['Y'],
                                                         test_X  = test.drop([classifier._name, 'Y'], axis=1),
                                                         test_Y  = test['Y'])
            classifier._prob_train = train[classifier._name]
            classifier._prob_test  = test [classifier._name]
        return classifier

    def SaveTo(self, saveto, save_test_train_data=True):
        import joblib
        if not os.path.isdir(saveto): os.mkdir(saveto)
        joblib.dump(self._model, os.path.join(saveto, self._name + '.model'))

        if save_test_train_data:
            with pd.HDFStore(os.path.join(saveto, self._name + '.model.h5'), 'w') as f:
                train = pd.concat([self._dataloader._train_X,
                                   self._dataloader._train_Y,
                                   pd.Series(data=self._prob_train, name=self._name)],
                                  axis=1)
                train.to_hdf(f, 'train')

                test = pd.concat([self._dataloader._test_X,
                                  self._dataloader._test_Y,
                                  pd.Series(data=self._prob_test, name=self._name)],
                                 axis=1)
                test.to_hdf(f, 'test' )
            
    def Train(self, saveto=None, verbose=True, save_test_train_data=True):
        from sklearn.metrics import roc_auc_score, classification_report
        from scipy.stats import ks_2samp
        
        self._model.fit(self._dataloader._train_X[self._dataloader._feature_vars], self._dataloader._train_Y)

        pred_train_y = self._model.predict(self._dataloader._train_X[self._dataloader._feature_vars])
        pred_test_y  = self._model.predict(self._dataloader._test_X [self._dataloader._feature_vars])
        self._prob_train = self._prob_func(self._model, self._dataloader._train_X[self._dataloader._feature_vars])
        self._prob_test  = self._prob_func(self._model, self._dataloader._test_X [self._dataloader._feature_vars])

        ntrain_sig, _ = np.histogram(self._prob_train[self._dataloader._train_Y == 1],
                                     bins=50, range=tuple(self._range), density=True)
        ntest_sig , _ = np.histogram(self._prob_test[self._dataloader._test_Y == 1],
                                     bins=50, range=tuple(self._range), density=True)
        
        ntrain_bkg, _ = np.histogram(self._prob_train[self._dataloader._train_Y == 0],
                                     bins=50, range=tuple(self._range), density=True)
        ntest_bkg , _ = np.histogram(self._prob_test[self._dataloader._test_Y == 0],
                                     bins=50, range=tuple(self._range), density=True)

        _, ksp_sig = ks_2samp(ntrain_sig, ntest_sig)
        _, ksp_bkg = ks_2samp(ntrain_bkg, ntest_bkg)
        
        training_classification_report = classification_report(pred_train_y, self._dataloader._train_Y,
                                                               target_names=['Background', 'Signal'],
                                                               output_dict=True)
        training_classification_report['Signal'    ]['KS-pvalue'] = ksp_sig
        training_classification_report['Background']['KS-pvalue'] = ksp_bkg
        training_classification_report['roc_auc'] = roc_auc_score(self._dataloader._train_Y,
                                                                  self._prob_train)


        testing_classification_report = classification_report(pred_test_y, self._dataloader._test_Y,
                                                              target_names=['Background', 'Signal'],
                                                              output_dict=True)
        testing_classification_report['Signal'    ]['KS-pvalue'] = ksp_sig
        testing_classification_report['Background']['KS-pvalue'] = ksp_bkg
        testing_classification_report['roc_auc'] = roc_auc_score(self._dataloader._test_Y,
                                                                 self._prob_test)

        

        if verbose:
            print('---------- training sample -------------')
            print(training_classification_report)
              
            print('----------- testing sample -------------')
            print(testing_classification_report)

        if saveto: self.SaveTo(saveto, save_test_train_data)
        return training_classification_report, testing_classification_report
        
    def Eval(self, X):
        return self._prob_func(self._model, X)

    def PlotROC(self, plots):
        from sklearn.metrics import roc_curve, auc
        import matplotlib.pyplot as plt
        
        train_fpr, train_tpr, _ = roc_curve(self._dataloader._train_Y, self._prob_train)
        test_fpr , test_tpr , _ = roc_curve(self._dataloader._test_Y , self._prob_test )

        train_auc = auc(train_fpr, train_tpr)
        test_auc  = auc(test_fpr , test_tpr )
        
        plt.plot(train_fpr, train_tpr, label='Train (auc = {:.2})'.format(test_auc), color='lightsteelblue')
        plt.plot(test_fpr , test_tpr , label='Test  (auc = {:.2})'.format(test_auc), color='royalblue'     )

        plt.plot([0, 1], [0, 1], '--', color=(0.6, 0.6, 0.6))
        plt.xlim([-0.05, 1.05])
        plt.ylim([-0.05, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver operating characteristic')
        plt.legend(loc="lower right")
        plt.grid()

        plt.savefig(os.path.join(plots, 'roc.pdf'))
        print('Wrote {}'.format(os.path.join(plots, 'roc.pdf')))
        plt.close()

    def PlotInputVariableLinearCorrelations(self, plots):
        import matplotlib.pyplot as plt
        
        s_matrix = self._dataloader._train_X[self._dataloader._feature_vars][self._dataloader._train_Y==1].corr()
        b_matrix = self._dataloader._train_X[self._dataloader._feature_vars][self._dataloader._train_Y==0].corr()

        fig, ax = plt.subplots()
        s_heatmap = ax.pcolor(s_matrix,
                              cmap=plt.get_cmap('RdBu'),
                              vmin=-1,
                              vmax=1)
        plt.colorbar(s_heatmap, ax=ax)

        ax.set_title('Signal Correlations')
        ax.set_xticks(np.arange(len(self._dataloader._feature_vars))+0.5, minor=False)
        ax.set_yticks(np.arange(len(self._dataloader._feature_vars))+0.5, minor=False)
        ax.set_xticklabels(self._dataloader._feature_vars, minor=False, ha='right', rotation=70)
        ax.set_yticklabels(self._dataloader._feature_vars, minor=False)

        for (i, j), z in np.ndenumerate(s_matrix):
            ax.text(i+0.5, j+0.5, '{:0.3f}'.format(z), ha='center', va='center', color='black' if abs(z)<0.75 else 'white')
        
        plt.tight_layout()
        name = os.path.join(plots, 'corr_s.pdf')
        fig.savefig(name)
        print('Wrote', name)
        plt.close()

        fig, ax = plt.subplots()
        b_heatmap = ax.pcolor(b_matrix,
                              cmap=plt.get_cmap('RdBu'),
                              vmin=-1,
                              vmax=1)
        plt.colorbar(b_heatmap, ax=ax)

        ax.set_title('Background Correlations')
        ax.set_xticks(np.arange(len(self._dataloader._feature_vars))+0.5, minor=False)
        ax.set_yticks(np.arange(len(self._dataloader._feature_vars))+0.5, minor=False)
        ax.set_xticklabels(self._dataloader._feature_vars, minor=False, ha='right', rotation=70)
        ax.set_yticklabels(self._dataloader._feature_vars, minor=False)

        for (i, j), z in np.ndenumerate(s_matrix):
            ax.text(i+0.5, j+0.5, '{:0.3f}'.format(z), ha='center', va='center', color='black' if abs(z)<0.75 else 'white')
        
        plt.tight_layout()
        name = os.path.join(plots, 'corr_b.pdf')
        fig.savefig(name)
        print('Wrote', name)
        plt.close()

    def PlotInputVariableDistributions(self, plots):
        import matplotlib.pyplot as plt

        for col in self._dataloader._feature_vars + self._dataloader._spectator_vars:
            minx  = self._dataloader._train_X[col].min()
            maxx  = self._dataloader._train_X[col].max()
            nbins = 50
            plt.hist(self._dataloader._train_X[col][self._dataloader._train_Y==1],
                     bins=nbins,
                     range=(minx, maxx),
                     alpha=0.7,
                     color='b',
                     density=True,
                     histtype='stepfilled',
                     label='Signal')
            plt.hist(self._dataloader._train_X[col][self._dataloader._train_Y==1],
                     bins=nbins,
                     range=(minx, maxx),
                     color='b',
                     density=True,
                     histtype='step')

            plt.hist(self._dataloader._train_X[col][self._dataloader._train_Y==0],
                     bins=nbins,
                     range=(minx, maxx),
                     color='r',
                     density=True,
                     hatch='//',
                     histtype='step',
                     label='Background')
            plt.hist(self._dataloader._train_X[col][self._dataloader._train_Y==0],
                     bins=nbins,
                     range=(minx, maxx),
                     color='r',
                     density=True,
                     histtype='step')

            plt.xlabel(col)
            plt.ylabel('Arbitrary Units')
            plt.legend(loc='best')
            name = os.path.join(plots, '{}_var_{}.pdf'.format('feature' if col in self._dataloader._feature_vars else 'spectator',
                                                              col))
            plt.savefig(name)
            print('Wrote', name)
            plt.close()

    def PlotCompareTrainTest(self, plots):
        import matplotlib.pyplot as plt
        nbins = 30

        # training background
        plt.hist(self._prob_train[self._dataloader._train_Y==0],
                 edgecolor='r',
                 hatch='/',
                 density=True,
                 range=tuple(self._range),
                 bins=nbins,
                 label='B (Train)',
                 histtype='step')
        
        # testing background
        hist, bins = np.histogram(self._prob_test[self._dataloader._test_Y==0],
                                  range=tuple(self._range), bins=nbins, density=True)    
        width = bins[1] - bins[0]
        center = (bins[:-1] + bins[1:]) / 2

        plt.errorbar(center, hist, xerr=width/2, fmt='.', c='r', label='B (Test)')

        # training signal
        plt.hist(self._prob_train[self._dataloader._train_Y==1],
                 color='b',
                 alpha=0.7,
                 density=True,
                 range=tuple(self._range),
                 bins=nbins,
                 label='S (Train)',
                 histtype='stepfilled')

        # testing background
        hist, bins = np.histogram(self._prob_test[self._dataloader._test_Y==1],
                                  range=tuple(self._range), bins=nbins, density=True)    
        width = bins[1] - bins[0]
        center = (bins[:-1] + bins[1:]) / 2

        plt.errorbar(center, hist, xerr=width/2, fmt='.', c='b', label='S (Test)')

        plt.xlabel(self._name)
        plt.ylabel('Arbitrary Units')
        plt.legend(loc='best')
        
        plt.savefig(os.path.join(plots, 'compare_train_test.pdf'))
        print('Wrote {}'.format(os.path.join(plots, 'compare_train_test.pdf')))

        plt.close()

class BDTClassifier(SKClassifier):
    def __init__(self,
                 name,
                 max_depth=3,
                 ada_boost_beta=0.5,
                 ntrees=800,
                 algorithm='SAMME',
                 min_node_size=0.05):
        from sklearn.tree import DecisionTreeClassifier
        from sklearn.ensemble import AdaBoostClassifier

        dt = DecisionTreeClassifier(max_depth        = max_depth,
                                    min_samples_leaf = min_node_size)
        bdt = AdaBoostClassifier(dt,
                                 algorithm     = algorithm,
                                 n_estimators  = ntrees,
                                 learning_rate = ada_boost_beta)
        def signal_proba(model, X):
            return AdaBoostClassifier.predict_proba(model, X)[:,1]
        
        self._prob_func = signal_proba
        super().__init__(name, bdt)
        
        self._range = [0,1]


    @staticmethod
    def LoadFrom(file_name, name):
        return SKClassifier.LoadFrom(file_name, BDTClassifier(name))
    @staticmethod
    def Reader(file_name, name):
        return SKClassifier.Reader(file_name, BDTClassifier(name))
        
class NBClassifier(SKClassifier):        
    def __init__(self, name, **kwargs):
        import sklearn.naive_bayes as nb

        def signal_proba(model, X):
            return nb.GaussianNB.predict_proba(model, X)[:,1]
        self._prob_func = signal_proba

        self._range = [0,1]
        
        super().__init__(name, nb.GaussianNB(**kwargs))

    @staticmethod
    def LoadFrom(file_name, name):
        return SKClassifier.LoadFrom(file_name, NBClassifier(name))
    @staticmethod
    def Reader(file_name, name):
        return SKClassifier.Reader(file_name, NBClassifier(name))


class MLPClassifier(SKClassifier):
    def __init__(self, name, **kwargs):
        import sklearn.neural_network
        
        def signal_proba(model, X):
            return sklearn.neural_network.MLPClassifier.predict_proba(model, X)[:,1]
        self._prob_func = signal_proba
        
        self._range = [0,1]
        
        super().__init__(name, sklearn.neural_network.MLPClassifier(**kwargs))
        
    @staticmethod
    def LoadFrom(file_name, name):
        return SKClassifier.LoadFrom(file_name, MLPClassifier(name))
    @staticmethod
    def Reader(file_name, name):
        return SKClassifier.Reader(file_name, MLPClassifier(name))

class GPClassifier(SKClassifier):
    def __init__(self, name, **kwargs):
        from sklearn.gaussian_process import GaussianProcessClassifier
        def signal_proba(model, X):
            return GaussianProcessClassifier.predict_proba(model, X)[:,1]
        self._prob_func = signal_proba
        self._range = [0,1]
        super().__init__(name, GaussianProcessClassifier(copy_X_train=False,
                                                         **kwargs))
    @staticmethod
    def LoadFrom(file_name, name):
        return SKClassifier.LoadFrom(file_name, GPClassifier(name))
    @staticmethod
    def Reader(file_name, name):
        return SKClassifier.Reader(file_name, GPClassifier(name))

class KNNClassifier(SKClassifier):
    def __init__(self, name, **kwargs):
        from sklearn.neighbors import KNeighborsClassifier
        def signal_proba(model, X):
            return KNeighborsClassifier.predict_proba(model, X)[:,1]
        self._prob_func = signal_proba
        self._range = [0,1]
        super().__init__(name, KNeighborsClassifier(**kwargs))
        
    @staticmethod
    def LoadFrom(file_name, name):
        return SKClassifier.LoadFrom(file_name, KNNClassifier(name))
    @staticmethod
    def Reader(file_name, name):
        return SKClassifier.Reader(file_name, KNNClassifier(name))

class RNClassifier(SKClassifier):
    def __init__(self, name, **kwargs):
        from sklearn.neighbors import RadiusNeighborsClassifier
        def signal_proba(model, X):
            return RadiusNeighborsClassifier.predict_proba(model, X)[:,1]
        self._prob_func = signal_proba
        self._range = [0,1]
        super().__init__(name, RadiusNeighborsClassifier(**kwargs))
        
    @staticmethod
    def LoadFrom(file_name, name):
        return SKClassifier.LoadFrom(file_name, RNClassifier(name))
    @staticmethod
    def Reader(file_name, name):
        return SKClassifier.Reader(file_name, RNClassifier(name))

_Types = {'bdt'     : BDTClassifier,
          'gaussnb' : NBClassifier ,
          'mlp'     : MLPClassifier,
          'gp'      : GPClassifier,
          'knn'     : KNNClassifier,
          'rn'      : RNClassifier}

def ClassifierType(model_type):
    return _Types[model_type]

class ManyClassifiers:
    def __init__(self, name):
        self._name = name
        self._models = {}
        self._dataloader = None
        
    def AddDataLoader(self, dataloader):
        self._dataloader = dataloader
        

    def BookModel(self, model):
        self._models[model._name] = model

    def TrainAll(self, *args, **kwargs):
        if not self._dataloader:
            print('A DataLoader has not been added')
        for model in self._models:
            self._models[model].AddDataLoader(self._dataloader)
            
        self._training_reports = {}
        self._testing_reports = {}
        for model in self._models:            
            print('Training %s...%s' % (model, type(self._models[model])))
            print(model)
            self._training_reports[model], self._testing_reports[model] = self._models[model].Train(saveto=os.path.join(self._name, model), *args, **kwargs)
            print()
            
    def EvaluateAll(self, sort_key = lambda report: (report['roc_auc'], report['Signal']['KS-pvalue'])):
        
        sorted_testing_reports = dict(sorted(self._testing_reports.items(),
                                             key=lambda report: sort_key(report[1]),
                                             reverse=True))

        print('Ranking: ')
        for i, model in enumerate(sorted_testing_reports):
            print(i+1, model, sort_key(sorted_testing_reports[model]))

        os.makedirs(self._name, exist_ok=True)
        for model in sorted_testing_reports:
            plot_dump = os.path.join(self._name, model, 'plots')
            os.makedirs(plot_dump, exist_ok=True)
            
            self._models[model].PlotROC             (plot_dump)
            self._models[model].PlotCompareTrainTest(plot_dump)
    
    def PlotAllInputVariableDistributions(self):
        for model in self._models:
            self._models[model].PlotInputVariableDistributions(os.path.join(self._name, model, 'plots'))

    
    def PlotAllInputVariableLinearCorrelations(self):
        for model in self._models:
            self._models[model].PlotInputVariableLinearCorrelations(os.path.join(self._name, model, 'plots'))

                       
