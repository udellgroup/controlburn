import warnings
import sklearn
from sklearn.tree import DecisionTreeRegressor
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import Lasso
from sklearn.linear_model import lasso_path
from sklearn.model_selection import KFold
import numpy as np
import pandas as pd
import mosek
import cvxpy as cp
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)

class ControlBurnClassifier:

    # Class attributes to store results.
    forest = []
    weights = []
    subforest = []

    #Class attributes for parameters to determine convergence.
    threshold = 10**-3
    tail = 5

    # Private Helper Methods
    def __log_odds_predict(self,X,log_odds_init,tree_list):
        """ Private helper method to return the prediction of a bag-boosted forest
        by summing the average log odds over boosting iterations. Returns the final
        array of log odds prediction.
        """
        res = []
        for i in tree_list:
            depth = i.max_depth
            pred = i.predict(X)
            res.append([depth,pred])
        res = pd.DataFrame(res,columns = ['depth','pred'])
        res = res.groupby('depth')['pred'].apply(np.mean).reset_index()
        res = np.sum(res['pred'].to_numpy()) + log_odds_init
        return res

    def __converge_test(self,sequence, threshold,tail_length):
        """ Private helper method to determine if a sequence converges. Sequence
        converges if the tail of the sequence falls within threshold of each
        other. Returns True if the sequence converged, False otherwise.
        """
        diff = np.diff(sequence)
        if len(diff) < (tail_length+1):
            return False
        else:
            return (max(np.abs(diff[-tail_length:])) < threshold)

    def __check_OOB_convergence(self,OOB_error_list):
        """ Private helper method to check if the improvement in out-of-bag error
        for a bag-boosted ensemble converges.  Returns True if the last element in the
        sequence of errors deltas is <=0, False otherwise.
        """
        if OOB_error_list[-1] <= 0:
            return True
        elif (len(OOB_error_list) < max(self.tail-2,1)+1):
            return False
        elif all([x < self.threshold for x in OOB_error_list[-max(self.tail-2,1):]]):
            return True
        else:
            return False

    #Forest Growing Methods
    def bag_forest(self,X,y):
        """ Forest growing algorithm that uses the class attribute max_depth as
        a hyperparameter.
        Adds trees of increasing depth to a bagged ensemble until max_depth is
        reached. The number of trees to add at each depth level is determined by
        checking if the training error converges.
        """

        self.X = X
        self.y = y
        threshold = self.threshold
        tail = self.tail
        train = X.copy()
        train = train.reset_index().drop('index',axis = 1)
        train['y'] = list(y)
        features = X.columns
        tree_list = []
        max_depth = self.max_depth

        for depth in range (1,max_depth+1):
            early_stop_pred = []
            early_stop_train_err = []
            converged = False

            while converged == False:
                train1 = train.sample(n = len(train), replace = True)
                y1 = train1['y']
                X1 = train1[features]
                clf = DecisionTreeClassifier(max_depth = depth)
                clf.fit(X1,y1)
                tree_list.append(clf)
                pred = clf.predict_proba(X[features])[:,1]
                early_stop_pred.append(pred)
                early_stop_train_err.append(sklearn.metrics.roc_auc_score(y,(np.mean(early_stop_pred,axis = 0))))
                converged = self.__converge_test(early_stop_train_err,threshold,tail)

        self.forest = tree_list
        return

    def bagboost_forest(self,X,y):
        """ Bag-boosting forest growing algorithm, no hyperparameters needed. The number of
        trees to grow at each boosting iteration is determined by the convergence of
        the training error. Out-of-bag error is used to determine how many boosting iterations to
        conduct.
        """
        threshold = self.threshold
        tail = self.tail
        self.X = X
        self.y = y
        y = pd.Series(y)

        X = X.reset_index().drop('index',axis = 1)
        y.index = X.index

        #initialization
        log_odds = np.log(sum(y)/(len(y)- sum(y)))
        prob = np.exp(log_odds)/(1+np.exp(log_odds))
        residual = y - prob

        train = X.copy()
        train['y'] = list(residual)
        features = X.columns
        pred_train = np.zeros(len(residual))
        tree_list = []

        OOB_error_list = []
        OOB_converged = False
        depth = 1

        while OOB_converged == False:
            early_stop_pred = []
            early_stop_train_err = []
            converged = False
            OOB_matrix = []
            tree_list1 = []

            if len(tree_list) > 0:
                current_pred = self.__log_odds_predict(X,log_odds,tree_list)
                X['current_pred'] = current_pred
                current_pred = X['current_pred']
                X.drop('current_pred',axis = 1,inplace = True)

            else:
                X['current_pred'] = log_odds
                current_pred = X['current_pred']
                X.drop('current_pred',axis = 1,inplace = True)

            while converged == False:
                train1 = train.sample(n = len(train), replace = True)
                OOB = train[~train.index.isin(train1.drop_duplicates().index.values)].index.values
                OOB_row = np.repeat(False,len(X))
                OOB_row[OOB] = True
                OOB_matrix.append(OOB_row)
                y1 = train1['y']
                X1 = train1[features]
                tree = DecisionTreeRegressor(max_depth = depth)
                tree.fit(X1,y1)
                tree_list.append(tree)
                tree_list1.append(tree)
                pred = tree.predict(X[features])
                early_stop_pred.append(pred)
                temp_pred = current_pred + (np.mean(early_stop_pred,axis = 0))
                temp_prob = np.exp(temp_pred)/(1+np.exp(temp_pred))
                early_stop_train_err.append(sklearn.metrics.roc_auc_score(y,temp_prob))
                converged = self.__converge_test(early_stop_train_err,threshold,tail)
                pred_train = pred_train + np.mean(early_stop_pred,axis = 0)
                if converged == False:
                    pred_train = pred_train - np.mean(early_stop_pred,axis = 0)

            indicators = pd.DataFrame(OOB_matrix).transpose()
            OOB_pred_list = []
            y2 = y.copy()

            y2 = y2[indicators.sum(axis = 1) > 0]
            current_pred = current_pred[indicators.sum(axis = 1) > 0]
            pred_matrix = np.array([tree_temp.predict(X) for tree_temp in tree_list1])
            ind_matrix = np.array(~indicators.values).transpose()
            masked = np.ma.masked_array(pred_matrix,ind_matrix)
            OOB_pred_list = masked.mean(axis = 0).data[indicators.sum(axis = 1) > 0]

            next_pred = np.array(current_pred) + np.array(OOB_pred_list)
            current_prob =  np.exp(current_pred)/(1+np.exp(current_pred))
            next_prob =  np.exp(next_pred)/(1+np.exp(next_pred))
            current_err = 1 - sklearn.metrics.roc_auc_score(y2,current_prob)
            next_err = 1 - sklearn.metrics.roc_auc_score(y2,next_prob)
            OOB_error_list.append(current_err-next_err)
            all_pred = self.__log_odds_predict(X,log_odds,tree_list)
            all_prob = np.exp(all_pred)/(1+np.exp(all_pred))
            train['y'] = y-all_prob
            OOB_converged = self.__check_OOB_convergence(OOB_error_list)
            depth = depth + 1

        self.forest = tree_list
        return

    def double_bagboost_forest(self,X,y):
        """ double bag-boosting forest growing algorithm, no hyperparameters needed. The number of
        trees to grow at each boosting iteration is determined by the convergence of
        the training error. Out-of-bag error is used to determine how many boosting iterations to
        conduct.
        """
        threshold = self.threshold
        tail = self.tail
        self.X = X
        self.y = y
        y = pd.Series(y)
        X = X.reset_index().drop('index',axis = 1)
        y.index = X.index

        #initialization
        log_odds = np.log(sum(y)/(len(y)- sum(y)))
        prob = np.exp(log_odds)/(1+np.exp(log_odds))
        residual = y - prob

        train = X.copy()
        train['y'] = list(residual)
        features = X.columns
        pred_train = np.zeros(len(residual))

        tree_list = []
        OOB_error_list = []
        OOB_converged = False

        depth = 1
        current_err = None
        depth_check = False
        depth_err = 99999
        depth_converged = False

        while depth_converged == False:

            early_stop_pred = []
            early_stop_train_err = []
            converged = False
            OOB_matrix = []
            tree_list1 = []

            if len(tree_list) > 0:
                current_pred = self.__log_odds_predict(X,log_odds,tree_list)
                X['current_pred'] = current_pred
                current_pred = X['current_pred']
                X.drop('current_pred',axis = 1,inplace = True)

            else:
                X['current_pred'] = log_odds
                current_pred = X['current_pred']
                X.drop('current_pred',axis = 1,inplace = True)

            index = 0
            while converged == False:
                train1 = train.sample(n = len(train), replace = True)
                OOB = train[~train.index.isin(train1.drop_duplicates().index.values)].index.values
                OOB_row = np.repeat(False,len(X))
                OOB_row[OOB] = True
                OOB_matrix.append(OOB_row)
                y1 = train1['y']
                X1 = train1[features]
                tree = DecisionTreeRegressor(max_depth = depth)
                tree.fit(X1,y1)
                tree_list.append(tree)
                index = index + 1
                tree_list1.append(tree)
                pred = tree.predict(X[features])
                early_stop_pred.append(pred)
                temp_pred = current_pred + (np.mean(early_stop_pred,axis = 0))
                temp_prob = np.exp(temp_pred)/(1+np.exp(temp_pred))
                early_stop_train_err.append(sklearn.metrics.roc_auc_score(y,temp_prob))
                converged = self.__converge_test(early_stop_train_err,threshold,tail)
                pred_train = pred_train + np.mean(early_stop_pred,axis = 0)
                if converged == False:
                    pred_train = pred_train - np.mean(early_stop_pred,axis = 0)

            indicators = pd.DataFrame(OOB_matrix).transpose()
            OOB_pred_list = []
            y2 = y.copy()

            y2 = y2[indicators.sum(axis = 1) > 0]
            current_pred = current_pred[indicators.sum(axis = 1) > 0]
            pred_matrix = np.array([tree_temp.predict(X) for tree_temp in tree_list1])
            ind_matrix = np.array(~indicators.values).transpose()
            masked = np.ma.masked_array(pred_matrix,ind_matrix)
            OOB_pred_list = masked.mean(axis = 0).data[indicators.sum(axis = 1) > 0]

            next_pred = np.array(current_pred) + np.array(OOB_pred_list)
            next_prob =  np.exp(next_pred)/(1+np.exp(next_pred))

            if current_err == None:
                current_prob =  np.exp(current_pred)/(1+np.exp(current_pred))
                current_err = 1 - sklearn.metrics.roc_auc_score(y2,current_prob)


            next_err = 1 - sklearn.metrics.roc_auc_score(y2,next_prob)
            OOB_error_list.append(current_err-next_err)
            OOB_converged = self.__check_OOB_convergence(OOB_error_list)

            if depth_check == True:
                depth_check = False
                if next_err > depth_err:
                    depth_converged = True

            if OOB_converged == True:
                tree_list = tree_list[:-index]
                index = 0
                depth = depth + 1
                depth_err = current_err
                depth_check = True

            current_err = next_err

            all_pred = self.__log_odds_predict(X,log_odds,tree_list)
            all_prob = np.exp(all_pred)/(1+np.exp(all_pred))
            train['y'] = y-all_prob

        self.forest = tree_list
        return


    #optional arguments
    max_depth = 10
    build_forest_method = bagboost_forest
    polish_method = RandomForestClassifier()
    alpha = 0.1
    solver = 'ECOS_BB'
    optimization_form= 'penalized'

    #initializer
    def __init__(self,alpha = 0.1,max_depth = 10, optimization_form= 'penalized',solver = 'ECOS_BB',build_forest_method = 'bagboost',
    polish_method = RandomForestClassifier(max_features = 'sqrt')):
        """
        Initalizes a ControlBurnClassifier object. Arguments: {alpha: regularization parameter, max_depth: optional
        parameter for incremental depth bagging, optimization_form: either 'penalized' or 'constrained', solver: cvxpy solver
        used to solve optimization problem, build_forest_method: either 'bagboost' or 'bag',polish_method: final model to
        fit on selected features}.
        """
        if optimization_form not in ['penalized','constrained']:
            raise ValueError("optimization_form must be either 'penalized' or 'constrained ")

        if build_forest_method not in ['bagboost','bag','doublebagboost']:
            raise ValueError("build_forest_method must be either 'bag', 'bagboost', or 'doublebagboost' ")

        if max_depth <= 0:
            raise ValueError("max_depth must be greater than 0")

        if alpha <= 0:
            raise ValueError("alpha must be greater than 0")

        self.alpha = alpha
        self.max_depth = max_depth
        self.optimization_form = optimization_form
        self.solver = solver
        self.polish_method = polish_method

        if  build_forest_method == 'bagboost':
            self.build_forest_method = self.bagboost_forest
        elif build_forest_method == 'bag':
            self.build_forest_method = self.bag_forest
        elif build_forest_method == 'doublebagboost':
            self.build_forest_method = self.double_bagboost_forest


    #Optimization Method

    def solve_lasso(self): #TODO: Finish
        if len(self.forest) == 0:
            raise Exception("Build forest first.")
        alpha = self.alpha
        X = self.X
        y = self.y
        y = pd.Series(y)
        y.index = X.index
        tree_list = self.forest
        pred = []
        ind = []

        if type(tree_list[0]) == sklearn.tree._classes.DecisionTreeClassifier:
            for tree in tree_list:
                pred.append(tree.predict_proba(X)[:,1])
                ind.append([int(x > 0) for x in tree.feature_importances_])
        else:
            for tree in tree_list:
                pred.append(tree.predict(X))
                ind.append([int(x > 0) for x in tree.feature_importances_])

        pred = np.transpose(np.array(pred,dtype = object))
        ind = np.transpose(ind)
        ind_vec=np.sum(ind,0)
        ind_vec = [x for x in ind_vec]
        inv_mat = np.linalg.inv(np.diag(ind_vec))
        transformed_matix=np.matmul(pred,inv_mat)

        sgd = SGDClassifier(loss = 'log',penalty = 'l1',
                    fit_intercept = True, alpha = alpha,
                    max_iter = 5000)

        sgd.fit(transformed_matix, y)
        weights = np.dot(inv_mat,sgd.coef_[0])
        self.weights = weights
        self.subforest = list(np.array(tree_list)[[w_ind != 0 for w_ind in list(weights)]])
        imp = []

        for i in range(0,len(weights)):
            imp.append(weights[i]*tree_list[i].feature_importances_)
        imp1 = np.sum(imp, axis = 0)
        self.feature_importances_ = imp1
        self.features_selected_ = list(np.array(X.columns)[[i != 0 for i in imp1]])
        return


    def solve_lasso_cvxpy(self):
        """ Solves LASSO optimization problem using class attribute alpha as the
        regularization parameter. Stores the selected features, weights, and
        subforest.
        """
        if len(self.forest) == 0:
            raise Exception("Build forest first.")
        alpha = self.alpha
        X = self.X
        y = self.y
        y = pd.Series(y)
        y.index = X.index
        tree_list = self.forest
        pred = []
        ind = []

        if type(tree_list[0]) == sklearn.tree._classes.DecisionTreeClassifier:
            for tree in tree_list:
                pred.append(tree.predict_proba(X)[:,1])
                ind.append([int(x > 0) for x in tree.feature_importances_])

        else:
            for tree in tree_list:
                pred.append(tree.predict(X))
                ind.append([int(x > 0) for x in tree.feature_importances_])

        pred = np.transpose(pred)
        ind = np.transpose(ind)
        w = cp.Variable(len(tree_list),nonneg=True)
        constraints = []

        if self.optimization_form == 'penalized':
            loss = -cp.sum( cp.multiply(y, pred@ w ) - cp.logistic(pred @ w) )
            objective = (1/len(y))*loss + alpha*cp.norm(cp.matmul(ind,w),1)


        if self.optimization_form == 'constrained':
            objective = -cp.sum(cp.multiply(y, pred@ w) - cp.logistic(pred @ w))
            constraints = [cp.norm(cp.matmul(ind,w),1)<= alpha]

        prob = cp.Problem(cp.Minimize(objective),constraints)

        if self.solver == 'MOSEK':
            prob.solve(solver = cp.MOSEK,mosek_params = {mosek.dparam.optimizer_max_time: 10000.0} )
        else:
            prob.solve(solver = self.solver)

        weights = np.asarray(w.value)
        weights[np.abs(weights) < self.threshold] = 0
        self.weights = weights
        self.subforest = list(np.array(tree_list)[[w_ind != 0 for w_ind in list(weights)]])
        imp = []

        for i in range(0,len(weights)):
            imp.append(weights[i]*tree_list[i].feature_importances_)
        imp1 = np.sum(imp, axis = 0)
        self.feature_importances_ = imp1
        self.features_selected_ = list(np.array(X.columns)[[i != 0 for i in imp1]])

        return

    #sklearn-api wrapper functions
    def fit(self,X,y):
        """ Wrapper function, builds a forest and solves LASSO optimization Problem
        to select a subforest. Trains final model on selected features.
        """
        self.build_forest_method(X,y)
        self.solve_lasso()
        if len(self.features_selected_) == 0:
            self.trained_polish = y
        else:
            self.trained_polish = self.polish_method.fit(X[self.features_selected_],y)

    def predict(self,X):
        """ Returns binary predictions of final model trained on selected features.
        """
        if len(self.features_selected_) == 0:
            return np.repeat(round(np.mean(self.trained_polish)),len(X))
        else:
            return self.trained_polish.predict(X[self.features_selected_])

    def predict_proba(self,X):
        """ Returns class probability predictions of final model trained on selected features.
        """
        if len(self.features_selected_) == 0:
            return np.repeat(np.mean(self.trained_polish),len(X))
        else:
            return self.trained_polish.predict_proba(X[self.features_selected_])

    def fit_transform(self,X,y):
        """ Returns dataframe of selected features.
        """
        self.build_forest_method(X,y)
        self.solve_lasso()
        if len(self.features_selected_) == 0:
            return pd.DataFrame()
        else:
            return X[self.features_selected_]

class ControlBurnRegressor:
    # Class attributes to store results.
    forest = []
    weights = []
    subforest = []
    alpha_range = []
    coef_path = []

    #Class attributes for parameters to determine convergence.
    threshold = 10**-3
    tail = 5

    # Private Helper Methods
    def __loss_gradient(self,y, y_hat):
        """ Helper function for gradient boosting, returns negative residuals
        """
        return -(y-y_hat)

    def __numpy_to_pandas(self,X):
        """ Helper function to convert numpy arrays to pandas DataFrame
        with fixed column names.
        """
        return pd.DataFrame(X,columns = ["f"+str(i) for i in range(X.shape[1])])

    def __converge_test(self,sequence, threshold,tail_length):
        """ Private helper method to determine if a sequence converges. Sequence
        converges if the tail of the sequence falls within threshold of each
        other. Returns True if the sequence converged, False otherwise.
        """
        diff = np.diff(sequence)
        if len(diff) < (tail_length+1):
            return False
        else:
            return (max(np.abs(diff[-tail_length:])) < threshold)

    def __check_OOB_convergence(self,OOB_error_list):
        """ Private helper method to check if the improvement in out-of-bag error
        for a bag-boosted ensemble converges.  Returns True if the last element in the
        sequence of errors deltas is <=0, False otherwise.
        """
        if OOB_error_list[-1] <= 0:
            return True
        elif (len(OOB_error_list) < max(self.tail-2,1)+1):
            return False
        elif all([x < self.threshold for x in OOB_error_list[-max(self.tail-2,1):]]):
            return True
        else:
            return False

    def __bag_boost_predict(self,X,tree_list):
        """ Private helper method to get he predictions from a bag-boosted
        ensemble.
        """
        res = []
        for i in tree_list:
            depth = i.max_depth
            pred = i.predict(X)
            res.append([depth,pred])
        res = pd.DataFrame(res,columns = ['depth','pred'])
        res = res.groupby('depth')['pred'].apply(np.mean).reset_index()
        res = np.sum(res['pred'].to_numpy())
        return res

    #Forest Growing Methods
    def bag_forest(self,X,y):
        """ Forest growing algorithm that uses the class attribute max_depth as
        a hyperparameter.
        Adds trees of increasing depth to a bagged ensemble until max_depth is
        reached. The number of trees to add at each depth level is determined by
        checking if the training error converges.
        """

        self.X = X
        self.y = y
        threshold = self.threshold
        tail = self.tail
        train = X.copy()
        train = train.reset_index().drop('index',axis = 1)
        train['y'] = list(y)
        features = X.columns
        tree_list = []
        max_depth = self.max_depth

        for depth in range (1,max_depth+1):
            early_stop_pred = []
            early_stop_train_err = []
            converged = False

            while converged == False:
                train1 = train.sample(n = len(train), replace = True)
                y1 = train1['y']
                X1 = train1[features]
                regressor = DecisionTreeRegressor(max_depth = depth)
                regressor.fit(X1,y1)
                tree_list.append(regressor)
                pred = regressor.predict(X[features])
                early_stop_pred.append(pred)
                early_stop_train_err.append(sklearn.metrics.mean_squared_error(y,(np.mean(early_stop_pred,axis = 0))))
                converged = self.__converge_test(early_stop_train_err,threshold,tail)

        self.forest = tree_list
        return

    def bagboost_forest(self,X,y):
        """ Bag-boosting forest growing algorithm, no hyperparameters needed. The number of
        trees to grow at each boosting iteration is determined by the convergence of
        the training error. Out-of-bag error is used to determine how many boosting iterations to
        conduct.
        """
        threshold = self.threshold
        tail = self.tail
        self.X = X
        self.y = y
        y = pd.Series(y)
        X = X.reset_index().drop('index',axis = 1)
        y.index = X.index
        pred_train = np.zeros(len(y))

        train = X.copy()
        train['y'] = list(y)
        features = X.columns

        tree_list = []
        OOB_error_list = []
        OOB_converged = False
        depth = 1

        while OOB_converged == False:

            early_stop_pred = []
            early_stop_train_err = []
            converged = False
            OOB_matrix = []
            tree_list1 = []

            if len(tree_list) > 0:
                current_pred = self.__bag_boost_predict(X,tree_list)
                X['current_pred'] = current_pred
                current_pred = X['current_pred']
                X.drop('current_pred',axis = 1,inplace = True)

            else:
                X['current_pred'] = np.mean(y)
                current_pred = X['current_pred']
                X.drop('current_pred',axis = 1,inplace = True)

            while converged == False:

                train1 = train.sample(n = len(train), replace = True)
                OOB = train[~train.index.isin(train1.drop_duplicates().index.values)].index.values
                OOB_row = np.repeat(False,len(X))
                OOB_row[OOB] = True
                OOB_matrix.append(OOB_row)
                y1 = train1['y']
                X1 = train1[features]
                tree = DecisionTreeRegressor(max_depth = depth)
                tree.fit(X1,y1)
                tree_list.append(tree)
                tree_list1.append(tree)
                pred = tree.predict(X[features])
                early_stop_pred.append(pred)
                pred_train = pred_train + np.mean(early_stop_pred,axis = 0)

                early_stop_train_err.append(sklearn.metrics.mean_squared_error(y,pred_train))
                converged = self.__converge_test(early_stop_train_err,threshold,tail)

                if converged == False:
                    pred_train = pred_train - np.mean(early_stop_pred,axis = 0)

            indicators = pd.DataFrame(OOB_matrix).transpose()
            OOB_pred_list = []
            y2 = y.copy()

            y2 = y2[indicators.sum(axis = 1) > 0]
            current_pred = current_pred[indicators.sum(axis = 1) > 0]
            pred_matrix = np.array([tree_temp.predict(X) for tree_temp in tree_list1])
            ind_matrix = np.array(~indicators.values).transpose()
            masked = np.ma.masked_array(pred_matrix,ind_matrix)
            OOB_pred_list = masked.mean(axis = 0).data[indicators.sum(axis = 1) > 0]

            next_pred = np.array(current_pred) + np.array(OOB_pred_list)
            current_err = sklearn.metrics.mean_squared_error(y2,current_pred)
            next_err = sklearn.metrics.mean_squared_error(y2,next_pred)

            OOB_error_list.append(current_err-next_err)

            residuals = -self.__loss_gradient(y, pred_train)
            train['y'] = residuals.values

            OOB_converged = self.__check_OOB_convergence(OOB_error_list)
            depth = depth + 1

        self.forest = tree_list
        return

    def double_bagboost_forest(self,X,y):
        """ Double bag-boosting forest growing algorithm, no hyperparameters needed. The number of
        trees to grow at each boosting iteration is determined by the convergence of
        the training error. Out-of-bag error is used to determine how many boosting iterations to
        conduct.
        """
        threshold = self.threshold
        tail = self.tail
        self.X = X
        self.y = y
        y = pd.Series(y)
        X = X.reset_index().drop('index',axis = 1)
        y.index = X.index
        pred_train = np.zeros(len(y))

        train = X.copy()
        train['y'] = list(y)
        features = X.columns

        tree_list = []
        OOB_error_list = []
        OOB_converged = False

        depth = 1
        current_err = None

        depth_check = False
        depth_err = 99999
        depth_converged = False

        while depth_converged == False:

            early_stop_pred = []
            early_stop_train_err = []
            converged = False
            OOB_matrix = []
            tree_list1 = []

            if len(tree_list) > 0:
                current_pred = self.__bag_boost_predict(X,tree_list)
                X['current_pred'] = current_pred
                current_pred = X['current_pred']
                X.drop('current_pred',axis = 1,inplace = True)

            else:
                X['current_pred'] = np.mean(y)
                current_pred = X['current_pred']
                X.drop('current_pred',axis = 1,inplace = True)

            index = 0

            while converged == False:

                train1 = train.sample(n = len(train), replace = True)
                OOB = train[~train.index.isin(train1.drop_duplicates().index.values)].index.values
                OOB_row = np.repeat(False,len(X))
                OOB_row[OOB] = True
                OOB_matrix.append(OOB_row)
                y1 = train1['y']
                X1 = train1[features]
                tree = DecisionTreeRegressor(max_depth = depth)
                tree.fit(X1,y1)
                tree_list.append(tree)
                index = index + 1
                tree_list1.append(tree)
                pred = tree.predict(X[features])
                early_stop_pred.append(pred)
                pred_train = pred_train + np.mean(early_stop_pred,axis = 0)

                early_stop_train_err.append(sklearn.metrics.mean_squared_error(y,pred_train))
                converged = self.__converge_test(early_stop_train_err,threshold,tail)

                if converged == False:
                    pred_train = pred_train - np.mean(early_stop_pred,axis = 0)

            indicators = pd.DataFrame(OOB_matrix).transpose()
            OOB_pred_list = []
            y2 = y.copy()

            y2 = y2[indicators.sum(axis = 1) > 0]
            current_pred = current_pred[indicators.sum(axis = 1) > 0]
            pred_matrix = np.array([tree_temp.predict(X) for tree_temp in tree_list1])
            ind_matrix = np.array(~indicators.values).transpose()
            masked = np.ma.masked_array(pred_matrix,ind_matrix)
            OOB_pred_list = masked.mean(axis = 0).data[indicators.sum(axis = 1) > 0]

            next_pred = np.array(current_pred) + np.array(OOB_pred_list)

            if current_err == None:
                current_err = sklearn.metrics.mean_squared_error(y2,current_pred)
            next_err = sklearn.metrics.mean_squared_error(y2,next_pred)


            OOB_error_list.append(current_err-next_err)
            OOB_converged = self.__check_OOB_convergence(OOB_error_list)

            if depth_check == True:
                depth_check = False
                if next_err > depth_err:
                    depth_converged = True

            if OOB_converged == True:
                tree_list = tree_list[:-index]
                index = 0
                depth = depth + 1
                depth_err = current_err
                depth_check = True


            current_err = next_err
            residuals = -self.__loss_gradient(y, pred_train)
            train['y'] = residuals.values

        self.forest = tree_list
        return


    #optional arguments
    max_depth = 10
    build_forest_method = bagboost_forest
    polish_method = RandomForestRegressor()
    alpha = 0.1
    solver = 'ECOS_BB'
    optimization_form= 'penalized'

    def skip_build_forest(self,X,y):
        self.X = X
        self.y = y
        return

    #initializer
    def __init__(self,alpha = 0.1,max_depth = 10, optimization_form= 'penalized',solver = 'ECOS_BB',build_forest_method = 'bagboost',
    polish_method = RandomForestRegressor(max_features = 'sqrt') , custom_forest = [] ):
        """
        Initalizes a ControlBurnClassifier object. Arguments: {alpha: regularization parameter, max_depth: optional
        parameter for incremental depth bagging, optimization_form: either 'penalized' or 'constrained', solver: cvxpy solver
        used to solve optimization problem, build_forest_method: either 'bagboost' or 'bag',polish_method: final model to
        fit on selected features}.
        """
        if optimization_form not in ['penalized','constrained']:
            raise ValueError("optimization_form must be either 'penalized' or 'constrained")

        if build_forest_method not in ['bagboost','bag','doublebagboost','custom']:
            raise ValueError("build_forest_method must be either 'bag', 'bagboost', 'doublebagboost', or 'custom' ")

        if max_depth <= 0:
            raise ValueError("max_depth must be greater than 0")

        if alpha <= 0:
            raise ValueError("alpha must be greater than 0")

        self.alpha = alpha
        self.max_depth = max_depth
        self.optimization_form = optimization_form
        self.solver = solver
        self.polish_method = polish_method
        if  build_forest_method == 'bagboost':
            self.build_forest_method = self.bagboost_forest
        elif build_forest_method == 'bag':
            self.build_forest_method = self.bag_forest
        elif build_forest_method == 'doublebagboost':
            self.build_forest_method = self.double_bagboost_forest

        elif build_forest_method == 'custom':
            if len(custom_forest) == 0:
                raise ValueError("Must provide external tree list")
            self.forest = custom_forest
            self.build_forest_method = self.skip_build_forest

    #Optimization Methods
    def solve_lasso_cvxpy(self):
        """ Solves LASSO optimization problem using class attribute alpha as the
        regularization parameter. Stores the selected features, weights, and
        subforest. Directly solves optimization problem directly via cvxpy.
        """
        if len(self.forest) == 0:
            raise Exception("Build forest.")
        alpha = self.alpha
        X = self.X
        y = self.y
        y = pd.Series(y)
        y.index = X.index
        tree_list = self.forest
        pred = []
        ind = []

        for tree in tree_list:
            pred.append(tree.predict(X))
            ind.append([int(x > 0) for x in tree.feature_importances_])

        pred = np.transpose(pred)
        ind = np.transpose(ind)
        w = cp.Variable(len(tree_list),nonneg=True)
        constraints = []

        if self.optimization_form == 'penalized':
            loss = cp.sum_squares(cp.matmul(pred,w)-y)
            objective = (1/len(y))*loss + alpha*cp.norm(cp.matmul(ind,w),1)

        if self.optimization_form == 'constrained':
            objective = cp.sum_squares(cp.matmul(pred,w)-y)
            constraints = [cp.norm(cp.matmul(ind,w),1)<= alpha]

        prob = cp.Problem(cp.Minimize(objective),constraints)

        if self.solver == 'MOSEK':
            prob.solve(solver = cp.MOSEK,mosek_params = {mosek.dparam.optimizer_max_time: 10000.0} )
        else:
            prob.solve(solver = self.solver)

        weights = np.asarray(w.value)
        weights[np.abs(weights) < self.threshold] = 0
        self.weights = weights
        self.subforest = list(np.array(tree_list)[[w_ind != 0 for w_ind in list(weights)]])
        imp = []

        for i in range(0,len(weights)):
            imp.append(weights[i]*tree_list[i].feature_importances_)
        imp1 = np.sum(imp, axis = 0)
        self.feature_importances_ = imp1
        self.features_selected_ = list(np.array(X.columns)[[i != 0 for i in imp1]])
        return


    def solve_lasso(self , costs = [], groups = [], sketching = 1.0):
        """ Solves LASSO optimization problem using class attribute alpha as the
        regularization parameter. Stores the selected features, weights, and
        subforest.
        """
        X = self.X
        if len(self.forest) == 0:
            raise Exception("Build forest.")

        if len(groups) >0:
            group_matrix = np.column_stack((np.arange(len(X.columns))
                                            ,groups,np.ones(len(X.columns))))
            group_matrix = pd.DataFrame(group_matrix,
                                        columns = ['feature','group','ind'])
            group_matrix = group_matrix.pivot_table(index = 'feature', columns = 'group',
                                values = 'ind').fillna(0).astype(int).values
            group_matrix = np.transpose(group_matrix)

        alpha = self.alpha
        if sketching < 1.0:
            X = self.X
            y = self.y
            nsample = int(np.floor(len(y)*sketching))
            to_sample = X.copy()
            to_sample['y'] = y
            to_sample = to_sample.sample(n = nsample)
            y = to_sample['y']
            X = to_sample.drop('y',axis = 1)
            y.index = X.index

        else:
            X = self.X
            y = self.y
            y = pd.Series(y)
            y.index = X.index

        tree_list = self.forest

        pred = []
        ind = []

        for tree in tree_list:
            pred.append(tree.predict(X))
            if (len(costs) == 0) & (len(groups) == 0):
                ind.append([int(x > 0) for x in tree.feature_importances_])

            elif (len(costs) != 0) & (len(groups) == 0):
                cost_matrix = np.diag(costs) #create diagonal cost matrix
                ind.append(np.dot(cost_matrix,
                            [int(x > 0) for x in tree.feature_importances_]))

            elif (len(costs) == 0) & (len(groups) != 0):
                ind.append((np.dot(group_matrix,
                [int(x > 0) for x in tree.feature_importances_])>0).astype(int))

        pred = np.transpose(pred)
        ind = np.transpose(ind)
        ind_vec=np.sum(ind,0)
        inv_mat = np.linalg.inv(np.diag(ind_vec))
        transformed_matix=np.matmul(pred,inv_mat)

        fit = Lasso(alpha = alpha,fit_intercept= False, positive = True).fit(transformed_matix,y)
        weights = np.matmul(inv_mat,np.transpose(fit.coef_))
        self.weights = weights
        self.subforest = list(np.array(tree_list)[[w_ind != 0 for w_ind in list(weights)]])

        imp = []
        for i in range(0,len(weights)):
            imp.append(weights[i]*tree_list[i].feature_importances_)
        imp1 = np.sum(imp, axis = 0)
        self.feature_importances_ = imp1
        self.features_selected_ = list(np.array(X.columns)[[i != 0 for i in imp1]])
        return


    def solve_l0(self, X, y, K, verbose = False):
        """ Selects the best subset of trees in the ensemble such that the
        total features used is equal to K. Best subset solver implemented using
        gurobi. Dependencies imported inside package since optional function.
        """

        import gurobipy as gp
        from gurobipy import GRB
        from itertools import product
        
        if len(self.forest) == 0:
            raise Exception("Build forest.")

        if type(X) == np.ndarray:
            X = self.__numpy_to_pandas(X)

        tree_list = self.forest
        self.X = X
        self.y = y

        M = len(tree_list)*2
        pred = []
        ind = []

        for tree in tree_list:
            pred.append(tree.predict(X))
            ind.append([int(x > 0) for x in tree.feature_importances_])

        Xpred = np.transpose(pred)
        ind = np.transpose(ind)
        response = np.array(y)

        regressor = gp.Model()
        if verbose == False:
            regressor.Params.LogToConsole = 0
        beta = regressor.addVars(len(tree_list), lb=0, name="beta")
        featurenonzero = regressor.addVars(len(X.columns),
                            vtype=GRB.BINARY, name="featurenonzero")

        Quad = np.dot(Xpred.T, Xpred)
        lin = np.dot(response.T, Xpred)
        obj = sum(0.5 * Quad[i,j] * beta[i] * beta[j] for i, j in product(range(len(tree_list)), repeat=2))
        obj -= sum(lin[i] * beta[i] for i in range(len(tree_list)))
        obj += 0.5 * np.dot(response, response)

        regressor.setObjective(obj, GRB.MINIMIZE)
        for i in range(0,len(X.columns)):
            regressor.addConstr(gp.quicksum(ind[i][j]*beta[j] for j in range(0,len(tree_list))) <= M*featurenonzero[i])
        regressor.addConstr(featurenonzero.sum() ==  K)

        regressor.optimize()
        weights = np.array([beta[i].X for i in range(len(tree_list))])
        subforest = np.array(tree_list)[weights != 0]
        features_selected = X.columns[[int(featurenonzero[i].X) == 1 for i in range(len(X.columns))]].values

        imp = []
        for i in range(0,len(weights)):
            imp.append(weights[i]*tree_list[i].feature_importances_)
        imp1 = np.sum(imp, axis = 0)

        self.weights = weights
        self.subforest = subforest
        self.feature_importances_= imp1
        self.features_selected_ = features_selected
        return



    def fit(self,X,y,costs = [], groups = [], sketching = 1.0):
        """ Wrapper function, builds a forest and solves LASSO optimization Problem
        to select a subforest. Trains final model on selected features.
        """
        if ( (round(np.mean(y))!= 0) | (round(np.std(y))!= 1)) :
            raise Exception("Please scale data before using ControlBurnRegressor.")

        if type(X) == np.ndarray:
            X = self.__numpy_to_pandas(X)

        self.build_forest_method(X,y)
        self.solve_lasso(costs, groups ,sketching = sketching)
        if len(self.features_selected_) == 0:
            self.trained_polish = y
        else:
            self.trained_polish = self.polish_method.fit(X[self.features_selected_],y)

    def predict(self,X):
        """ Returns predictions of final model trained on selected features.
        """
        if type(X) == np.ndarray:
            X = self.__numpy_to_pandas(X)

        if len(self.features_selected_) == 0:
            return np.repeat((np.mean(self.trained_polish)),len(X))
        else:
            return self.trained_polish.predict(X[self.features_selected_])


    def fit_transform(self,X,y):
        """ Returns dataframe of selected features.
        """
        if type(X) == np.ndarray:
            X = self.__numpy_to_pandas(X)

        self.build_forest_method(X,y)
        self.solve_lasso()
        if len(self.features_selected_) == 0:
            return pd.DataFrame()
        else:
            return X[self.features_selected_]

    def solve_lasso_path(self, X,y ,
                        n_alphas = 500, costs = [],
                         groups = [], kwargs = {}):
        """ Compute the entire lasso regularization path using warm start
        continuation. Returns a list of alphas and an numpy array of fitted
        coefficients.
        """
        if type(X) == np.ndarray:
            X = self.__numpy_to_pandas(X)

        if len(self.forest) == 0:
            raise Exception("Build forest.")

        if len(groups) >0:
            group_matrix = np.column_stack((np.arange(len(X.columns))
                                            ,groups,np.ones(len(X.columns))))
            group_matrix = pd.DataFrame(group_matrix,
                                        columns = ['feature','group','ind'])
            group_matrix = group_matrix.pivot_table(index = 'feature', columns = 'group',
                                values = 'ind').fillna(0).astype(int).values
            group_matrix = np.transpose(group_matrix)

        alpha = self.alpha
        y = pd.Series(y)
        y.index = X.index
        tree_list = self.forest

        pred = []
        ind = []

        for tree in tree_list:
            pred.append(tree.predict(X))
            if (len(costs) == 0) & (len(groups) == 0):
                ind.append([int(x > 0) for x in tree.feature_importances_])

            elif (len(costs) != 0) & (len(groups) == 0):
                cost_matrix = np.diag(costs) #create diagonal cost matrix
                ind.append(np.dot(cost_matrix,
                            [int(x > 0) for x in tree.feature_importances_]))

            elif (len(costs) == 0) & (len(groups) != 0):
                ind.append((np.dot(group_matrix,
                [int(x > 0) for x in tree.feature_importances_])>0).astype(int))

        pred = np.transpose(np.array(pred,dtype = object))
        ind = np.transpose(ind)

        ind_vec=np.sum(ind,0)
        inv_mat = np.linalg.inv(np.diag(ind_vec))
        transformed_matix=np.matmul(pred,inv_mat)

        alphas, coef_path, _ = lasso_path(transformed_matix, y,
                        n_alphas = n_alphas, positive = True, **kwargs)
        coef_path = np.transpose(coef_path)

        self.alpha_range = alphas
        self.coef_path = [np.dot(inv_mat,coef) for coef in coef_path]
        return alphas,self.coef_path

    ## TODO: complete this section
    def fit_cv(self,X,y, nfolds = 5, n_alphas = 500, show_plot = True, kwargs = {}):
        """ Compute the entire lasso path and select the best parameter using
            a nfold cross validation. Returns the best regularization parameter,
            support size, and selected features
        """
        if type(X) == np.ndarray:
            X = self.__numpy_to_pandas(X)

        if ( (round(np.mean(y))!= 0) | (round(np.std(y))!= 1)) :
            raise Exception("Please scale data before using ControlBurnRegressor.")

        alpha = self.alpha
        y = pd.Series(y)
        y.index = X.index


        kf = KFold(n_splits=nfolds)
        acc_all = np.array([])
        alphas_all = np.array([])
        feats_all = []

        for train_index, test_index in kf.split(X):
            xTrain1, xTest1 = X.iloc[train_index], X.iloc[test_index]
            yTrain1, yTest1 = y.iloc[train_index], y.iloc[test_index]
            self.build_forest_method(xTrain1,yTrain1)
            tree_list1 = self.forest
            pred = []
            ind = []

            for tree in tree_list1:
                pred.append(tree.predict(xTrain1))
                ind.append([int(f > 0) for f in tree.feature_importances_])

            pred = np.transpose(np.array(pred,dtype = object))
            ind = np.transpose(ind)

            ind_vec=np.sum(ind,0)
            inv_mat = np.linalg.inv(np.diag(ind_vec))
            transformed_matix=np.matmul(pred,inv_mat)
            alphas, coef_path, _ = lasso_path(transformed_matix, yTrain1 ,
                                        n_alphas = n_alphas, positive = True,**kwargs)
            coef = np.transpose(coef_path)

            acc = []
            alpha_list = []
            feats_list = []
            for i in range(0,len(coef)):
                #weights = np.dot(inv_mat,coef[i])
                weights = coef[i]
                ind = weights>0
                selected_ensemble= np.array(tree_list1)[ind]
                selected_weights = weights[ind]
                if len(selected_ensemble) > 0:
                    pred = 0
                    feats_used = np.array([])
                    for j in range(len(selected_ensemble)):
                        pred = pred + selected_ensemble[j].predict(xTest1)*selected_weights[j]
                        feats_used= np.append(feats_used,
                                    X.columns[selected_ensemble[j].feature_importances_ >0].values)

                    acc.append(sklearn.metrics.mean_squared_error(yTest1,pred))
                    alpha_list.append(alphas[i])
                    feats_all.append(list(set(feats_used)))

            acc_all = np.append(acc_all,acc)
            alphas_all = np.append(alphas_all,alpha_list)

        results = pd.DataFrame(np.column_stack((alphas_all,acc_all,
                        [len(x) for x in feats_all])),columns = ['alpha','accuracy','num_feats'])

        results = results.sort_values('alpha',ascending = True)
        results_agg = results.groupby('num_feats').agg(['mean','std']).reset_index()
        results_agg = results_agg[results_agg['accuracy']['mean']<1]
        best_nfeats = results_agg.sort_values(('accuracy','mean'),ascending = True)\
        .head(1)['num_feats'].values[0]

        best_feats = list(np.array(feats_all)\
                       [[len(x) == best_nfeats for x in feats_all]])

        best_feats = list(set([tuple(sorted(i)) for i in best_feats]))
        best_alpha = results.sort_values('accuracy',
                                        ascending = True)['alpha'].values[0]

        self.X = X
        self.Y = y
        self.alpha = best_alpha
        self.forest = []
        self.subforest = []
        self.weights = []
        self.fit(X,y)

        if show_plot == True:

            fig, (ax1,ax2) = plt.subplots(1,2,sharey = True , figsize = (12,5))
            results_plot = results[results['accuracy']<1]
            ax1.plot(results_plot['alpha'],results_plot['accuracy'],
                                                    color = 'blue',alpha = .5)
            ax1.set_xlabel('Regularization Parameter')
            ax1.set_ylabel('Validation Error')

            y_red_feats = results_agg.sort_values(('accuracy','mean'),ascending = True)\
            .head(1)['accuracy']['mean'].values

            results_agg = results.groupby('num_feats').agg(['mean','std']).reset_index()
            results_agg = results_agg[results_agg['accuracy']['mean']<1]

            ax2.errorbar(results_agg['num_feats'],\
                         results_agg['accuracy']['mean'],results_agg['accuracy']['std'],
                         alpha = 0.3, color = 'blue',zorder = 1)

            ax2.scatter(results_agg['num_feats'],results_agg['accuracy']\
                     ['mean'],color = 'blue')
            ax2.scatter(best_nfeats,y_red_feats,color = 'red',zorder = 2)
            ax2.set_xlabel('Number of Features Selected')
            plt.draw()


        return best_alpha, best_nfeats,best_feats
