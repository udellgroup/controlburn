
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import permutations

#TODO: InterpretClassifier class

class InterpretRegressor:
    cb = None
    X = None
    y = None

    #initializer
    def __init__(self, ControlBurnRegressor, X,y):
    """ Initalize an interpreter class object, requires a ControlBurnRegressor
    class to initalize.
    """
        self.cb = ControlBurnRegressor
        self.X =  X
        self.y = y


    def list_subforest(self, show_plot = False):
        """ Lists the features used in each tree of the selected subforest to
        give a sense of model structure, returns the array of features used
        """
        tree_list = self.cb.subforest
        cols = self.X.columns
        features_used = []
        for tree1 in tree_list:
            features_used.append(cols[tree1.feature_importances_ > 0].values)
            if show_plot == True:
                print(cols[tree1.feature_importances_ > 0].values)
        return features_used



    def plot_single_feature_shape(self, feature, show_plot = True):
        """ Plot a shape function that demonstrates the contribution of single
        feature trees on the response.
        """
        tree_list = self.cb.subforest
        cols = self.X.columns
        sub_weights = self.cb.weights[self.cb.weights>0]

        loc = 0
        pred_all = []
        for tree in tree_list:
            if ((feature in cols[tree.feature_importances_>0]) &   \
                (len(cols[tree.feature_importances_>0]) == 1)) :

                x_temp = pd.DataFrame(np.linspace(-1,1,1000),columns = [feature])

                for i in cols:
                    if i != feature:
                        x_temp[i] = 0
                x_temp = x_temp[cols]
                pred = tree.predict(x_temp)
                pred_all.append(pred*sub_weights[loc])
            loc = loc+1

        if len(pred_all)>0:
            pred_all = np.sum(pred_all,axis = 0)
            plt.figure()
            plt.plot(np.linspace(-1,1,1000),pred_all)
            plt.xlabel(feature)
            plt.ylabel('Contribution to prediction')
            plt.title('Shape function for feature: ' + feature)
            return np.column_stack((np.linspace(-1,1,1000),pred_all))

        else:
            print('No single feature trees found for feature: ' + feature)
            return None

    def plot_pairwise_interactions(self,feature1,feature2,show_plot = True):
        """Plot a heatmap showing impact of pairwise interaction on
        the response.
        """
        tree_list = self.cb.subforest
        cols = self.X.columns
        sub_weights = self.cb.weights[self.cb.weights>0]
        pairs = list(permutations(np.linspace(-3,3,200),2))
        x_temp = pd.DataFrame(pairs,columns = [feature1,feature2])

        for i in cols:
            if i not in [feature1,feature2]:
                x_temp[i] = 0
        x_temp = x_temp[cols]
        pred_all = []
        loc = 0
        for tree in tree_list:
            if ((feature1 in cols[tree.feature_importances_>0]) \
             &(feature2 in cols[tree.feature_importances_>0]) & \
                (len(cols[tree.feature_importances_>0]) == 2)):

                pred = tree.predict(x_temp)
                pred_all.append(pred*sub_weights[loc])
            loc = loc + 1
        pred_all = np.sum(pred_all,axis = 0)

        df = pd.DataFrame(pairs,columns = [feature1,feature2])
        df = df[[feature1, feature2]].round(3)
        df['contribution'] = pred_all
        df['contribution'] = df['contribution'].round(3)

        df_plot = df.pivot_table(index=feature2,
                    columns=feature1, values='contribution')
        if show_plot == True:
            heatmap = sns.heatmap(df_plot , cmap='RdYlGn_r'
                            , fmt='.4f', center = 0,
                            cbar_kws={'label': 'Contribution to prediction'})
            heatmap.invert_yaxis()
            plt.title('Pairwise interactions: ' + feature1 +', ' + feature2)
            #trim formating
            #color scale so 0 white
        return df

        #TODO: plotting shape function (combine) interpret.plot

    def plot_regularization_path(self, show_plot = True , more_colors = False):
        """ Plot the LASSO regularization path for a ControlBurnRegressor.
        Feature importance for a feature computed as the weighted sum of
        feature importances for each tree in the selected subforest.
        """
        if len(self.cb.coef_path) == 0:
            raise Exception("Please fit LASSO path before plotting.")

        alphas = self.cb.alpha_range
        coef_path = self.cb.coef_path
        feats = self.cb.X.columns
        tree_list = self.cb.forest

        results = []
        for i in range(0,len(coef_path)):
            weights = coef_path[i]
            ind = weights>0
            selected_ensemble= np.array(tree_list, dtype= object )[ind]
            selected_weights = weights[ind]
            feat_imp = list(np.repeat(0,len(feats)))

            if len(selected_ensemble) > 0:
                for i in range(0,len(selected_ensemble)):
                    learner = selected_ensemble[i]
                    w = selected_weights[i]
                    importances = list(w*learner.feature_importances_)
                    feat_imp.append(importances)
                feat_imp =  np.mean(feat_imp,axis = 0)
            results.append(feat_imp)

        results = pd.DataFrame(results,columns = feats)
        results['penalties'] = alphas

        if show_plot == True:
            fig = plt.figure(figsize = (12,9))
            ax1 = fig.add_subplot(111)
            if more_colors > 0:
                colormap = plt.cm.nipy_spectral
                ax1.set_prop_cycle(color = [colormap(i) \
                        for i in np.linspace(0, 1,more_colors)])
            for i in feats:
                ax1.plot(np.log(results['penalties']),results[i], label = i)
            ax1.legend()
            ax1.set_xlabel('Log Regularization Penalty')
            ax1.set_ylabel('Weighted Feature Importance')
            ax1.set_title('LASSO Regularization Path')

        return results
