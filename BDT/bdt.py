import numpy as np
from pandana import *
from PandAnaTools.BDT.dataloader import TrainTestDataLoader
import os
"""
Credit to https://betatim.github.io/posts/sklearn-for-TMVA-users/ 
around which this library is designed 

Author: Derek.Doyle@colostate.edu
"""

class BDT:
    """ 
    A wrapper class for scikit learn BDT libraries
    designed for those familiar with TMVA.

    Defaults correspond to default TMVA options
    """
    def __init__(self,
                 name,
                 max_depth=3,
                 ada_boost_beta=0.5,
                 ntrees=800,
                 algorithm='SAMME',
                 min_node_size=0.05):
        from sklearn.tree import DecisionTreeClassifier
        from sklearn.ensemble import AdaBoostClassifier

        self._name = name
        
        # DecisionTree min_samples_leaf = min_node_size * ntrain
        self._min_node_size = min_node_size
        # DecisionTree depth
        self._max_depth = 3

        
        # AdaBoost Learning Rate
        self._ada_boost_beta = ada_boost_beta
        # AdaBoost n_estimators
        self._ntrees = ntrees
        # AdaBoost Boosting algorithm
        self._algorithm = algorithm

        self._dt = DecisionTreeClassifier(max_depth        = self._max_depth,
                                          min_samples_leaf = self._min_node_size)
        self._bdt = AdaBoostClassifier(self._dt,
                                       algorithm     = self._algorithm,
                                       n_estimators  = self._ntrees,
                                       learning_rate = self._ada_boost_beta)
    
    def Train(self):
        from sklearn.metrics import roc_auc_score, classification_report
        
        self._bdt.fit(self._dataloader._train_X, self._dataloader._train_Y)

        pred_train_y = self._bdt.predict(self._dataloader._train_X)
        pred_test_y  = self._bdt.predict(self._dataloader._test_X )
        self._decision_train = pd.Series(data  = self._bdt.decision_function(self._dataloader._train_X),
                                         index = self._dataloader._train_X.index,
                                         name  = self._name)
        self._decision_test  = pd.Series(data  = self._bdt.decision_function(self._dataloader._test_X),
                                         index = self._dataloader._test_X.index,
                                         name  = self._name)

        print('---------- training sample -------------')
        print(classification_report(pred_train_y, self._dataloader._train_Y,
                                    target_names=['Background', 'Signal']))
        print('Area under ROC curve: {:.3f}'.format(roc_auc_score(self._dataloader._train_Y,
                                                                  self._decision_train)))
        print('----------- testing sample -------------')
        print(classification_report(pred_test_y, self._dataloader._test_Y,
                                             target_names=['Background', 'Signal']))
        print('Area under ROC curve: {:.3f}'.format(roc_auc_score(self._dataloader._test_Y,
                                                                  self._bdt.decision_function(self._dataloader._test_X))))

        self.SaveTo()
        
    def SaveTo(self):
        import joblib
        joblib.dump(self._bdt, self._name + '.bdt')

        with pd.HDFStore(self._name + '.bdt.h5', 'w') as f:
            train = pd.concat([self._dataloader._train_X,
                               self._dataloader._train_Y,
                               self._decision_train], axis=1)
            train.to_hdf(f, 'train')

            test = pd.concat([self._dataloader._test_X,
                              self._dataloader._test_Y,
                              self._decision_test], axis=1)
            test.to_hdf(f, 'test' )


    
    @staticmethod
    def Reader(file_name):
        import joblib
        bdt = BDT()
        bdt._bdt = joblib.load(file_name)
        return bdt

    @staticmethod
    def LoadFrom(file_name, name):
        bdt = BDT(name)
        with pd.HDFStore(file_name) as f:
            train = pd.read_hdf(f, 'train')
            test  = pd.read_hdf(f, 'test' )
            bdt._dataloader = TrainTestDataLoader(train_X = train.drop([name, 'Y'], axis=1),
                                                   train_Y = train['Y'],
                                                   test_X  = test.drop([name, 'Y'], axis=1),
                                                   test_Y  = test['Y'])
            bdt._decision_train = train[name]
            bdt._decision_test  = test [name]
        return bdt


    def AddDataLoader(self, dataloader):
        self._dataloader = dataloader


    def PlotROC(self, plots):
        from sklearn.metrics import roc_curve, auc
        import matplotlib.pyplot as plt
        
        train_fpr, train_tpr, _ = roc_curve(self._dataloader._train_Y, self._decision_train)
        test_fpr , test_tpr , _ = roc_curve(self._dataloader._test_Y , self._decision_test )

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
        
        s_matrix = self._dataloader._train_X[self._dataloader._train_Y==1].corr()
        b_matrix = self._dataloader._train_X[self._dataloader._train_Y==0].corr()

        fig, ax = plt.subplots()
        s_heatmap = ax.pcolor(s_matrix,
                              cmap=plt.get_cmap('RdBu'),
                              vmin=-1,
                              vmax=1)
        plt.colorbar(s_heatmap, ax=ax)

        ax.set_title('Signal Correlations')
        labels = self._dataloader._train_X.columns
        ax.set_xticks(np.arange(len(labels))+0.5, minor=False)
        ax.set_yticks(np.arange(len(labels))+0.5, minor=False)
        ax.set_xticklabels(labels, minor=False, ha='right', rotation=70)
        ax.set_yticklabels(labels, minor=False)

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
        labels = self._dataloader._train_X.columns
        ax.set_xticks(np.arange(len(labels))+0.5, minor=False)
        ax.set_yticks(np.arange(len(labels))+0.5, minor=False)
        ax.set_xticklabels(labels, minor=False, ha='right', rotation=70)
        ax.set_yticklabels(labels, minor=False)

        for (i, j), z in np.ndenumerate(s_matrix):
            ax.text(i+0.5, j+0.5, '{:0.3f}'.format(z), ha='center', va='center', color='black' if abs(z)<0.75 else 'white')
        
        plt.tight_layout()
        name = os.path.join(plots, 'corr_b.pdf')
        fig.savefig(name)
        print('Wrote', name)
        plt.close()
        

    def PlotInputVariableDistributions(self, plots):
        import matplotlib.pyplot as plt

        for col in self._dataloader._train_X.columns:
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
            name = os.path.join(plots, 'input_vars_{}.pdf'.format(col))
            plt.savefig(name)
            print('Wrote', name)
            plt.close()
            
    def PlotCompareTrainTest(self, plots):
        import matplotlib.pyplot as plt
        nbins = 30

        # training background
        plt.hist(self._decision_train[self._dataloader._train_Y==0],
                 edgecolor='r',
                 hatch='/',
                 density=True,
                 range=(-1,1),
                 bins=nbins,
                 label='B (Train)',
                 histtype='step')
        
        # testing background
        hist, bins = np.histogram(self._decision_test[self._dataloader._test_Y==0],
                                  range=(-1,1), bins=nbins, density=True)    
        width = bins[1] - bins[0]
        center = (bins[:-1] + bins[1:]) / 2

        plt.errorbar(center, hist, xerr=width/2, fmt='.', c='r', label='B (Test)')

        # training signal
        plt.hist(self._decision_train[self._dataloader._train_Y==1],
                 color='b',
                 alpha=0.7,
                 density=True,
                 range=(-1,1),
                 bins=nbins,
                 label='S (Train)',
                 histtype='stepfilled')

        # testing background
        hist, bins = np.histogram(self._decision_test[self._dataloader._test_Y==1],
                                  range=(-1,1), bins=nbins, density=True)    
        width = bins[1] - bins[0]
        center = (bins[:-1] + bins[1:]) / 2

        plt.errorbar(center, hist, xerr=width/2, fmt='.', c='b', label='S (Test)')

        plt.xlabel(self._name)
        plt.ylabel('Arbitrary Units')
        plt.legend(loc='best')
        
        plt.savefig(os.path.join(plots, 'compare_train_test.pdf'))
        print('Wrote {}'.format(os.path.join(plots, 'compare_train_test.pdf')))

        plt.close()
        

                                          
