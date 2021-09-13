import sklearn
from sklearn.tree import DecisionTreeRegressor
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
import numpy as np
import pandas as pd
import mosek
import cvxpy as cp

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
        start = time.perf_counter()
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
    polish_method = RandomForestClassifier
    alpha = 0.1
    solver = 'ECOS_BB'
    optimization_form= 'penalized'

    #initializer
    def __init__(self,alpha = 0.1,max_depth = 10, optimization_form= 'penalized',solver = 'ECOS_BB',build_forest_method = 'bagboost',
    polish_method = RandomForestClassifier):
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
    def solve_lasso(self):
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
            self.trained_polish = self.polish_method().fit(X[self.features_selected_],y)

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

    #Class attributes for parameters to determine convergence.
    threshold = 10**-3
    tail = 5

    # Private Helper Methods

    def __loss_gradient(self,y, y_hat):
        return -(y-y_hat)

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
    polish_method = RandomForestRegressor
    alpha = 0.1
    solver = 'ECOS_BB'
    optimization_form= 'penalized'

    #initializer
    def __init__(self,alpha = 0.1,max_depth = 10, optimization_form= 'penalized',solver = 'ECOS_BB',build_forest_method = 'bagboost',
    polish_method = RandomForestRegressor):
        """
        Initalizes a ControlBurnClassifier object. Arguments: {alpha: regularization parameter, max_depth: optional
        parameter for incremental depth bagging, optimization_form: either 'penalized' or 'constrained', solver: cvxpy solver
        used to solve optimization problem, build_forest_method: either 'bagboost' or 'bag',polish_method: final model to
        fit on selected features}.
        """
        if optimization_form not in ['penalized','constrained']:
            raise ValueError("optimization_form must be either 'penalized' or 'constrained")

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
    def solve_lasso(self):
        """ Solves LASSO optimization problem using class attribute alpha as the
        regularization parameter. Stores the selected features, weights, and
        subforest.
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

        #sklearn-api wrapper functions
    def fit(self,X,y):
        """ Wrapper function, builds a forest and solves LASSO optimization Problem
        to select a subforest. Trains final model on selected features.
        """
        if ( (round(np.mean(y))!= 0) | (round(np.std(y))!= 1)) :
            raise Exception("Please scale data before using ControlBurnRegressor.")

        self.build_forest_method(X,y)
        self.solve_lasso()
        if len(self.features_selected_) == 0:
            self.trained_polish = y
        else:
            self.trained_polish = self.polish_method().fit(X[self.features_selected_],y)

    def predict(self,X):
        """ Returns predictions of final model trained on selected features.
        """
        if len(self.features_selected_) == 0:
            return np.repeat((np.mean(self.trained_polish)),len(X))
        else:
            return self.trained_polish.predict(X[self.features_selected_])

    def fit_transform(self,X,y):
        """ Returns dataframe of selected features.
        """
        self.build_forest_method(X,y)
        self.solve_lasso()
        if len(self.features_selected_) == 0:
            return pd.DataFrame()
        else:
            return X[self.features_selected_]
