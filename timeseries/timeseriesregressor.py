from skmultiflow.core import BaseSKMObject, RegressorMixin
from skmultiflow.utils.data_structures import InstanceWindow
from skmultiflow.utils.utils import *

class TimeSeriesRegressor(BaseSKMObject, RegressorMixin):
    
    def __init__(self, estimator :RegressorMixin, max_window_size=100):
        super().__init__()
        
        if not isinstance(estimator, RegressorMixin):
            raise ValueError("estimator must be a Regressor, "
                                     "Call TimeSeriesRegressor with an instance of RegressorMixin")
        
        self.max_window_size = max_window_size
        self.estimator = estimator
        self.window = InstanceWindow(max_size=max_window_size, dtype=float)
        self.first_fit = True
    
    def partial_fit(self, X, y=None, sample_weight=None):
        """ Partially fits the model on the samples X and corresponding targets y.
        
        Parameters
        ----------
        X: Numpy.ndarray of shape (n_samples, n_features)
            The data upon which the algorithm will create its model. 
            If y is not provided the value X[t+1] in X will be used as target For X[t]
            
        y: numpy.ndarray, optional
            An array-like containing the targets for all samples in X.
            y must have the shape as X
            
        sample_weight: Not used.
        
        Returns
        -------
        TimeSeriesRegressor
            self
        Notes
        -----
        For the TimeSeries Classifier, fitting the model is the
        equivalent of inserting the newer samples in the observed window,
        and if the max_window_size is reached, removing older results and then using
        the the max_window_size past X values to predict future X values by feeding
        them as features to the provided model.
        To store the viewed samples we use a InstanceWindow object. For this 
        class' documentation please visit skmultiflow.core.utils.data_structures
        """
        if len(X.shape) == 1:
            X = X.reshape(-1,1)
        if len(y.shape) == 1:
            y = y.reshape(-1,1)
        r = X.shape[0]
        
        if y is not None:
            r_t = y.shape[0]

            if r != r_t:
                raise ValueError("Batch size of X is different from the number of attributes in y "
                                     "Batch size of must be the same for X and y")
                
        if self.first_fit:
            if r <= self.max_window_size:
                raise ValueError("Number of elments of First call to partial_fit less than max_window_size "
                                     "Call partial_fit with more than {} elements".format(self.max_window_size))

        for i in range(r):
            if y is not None:
                self.window.add_element(np.asarray([X[i]]), np.asarray([y[i]]))
            elif i > 0:
                self.window.add_element(np.asarray([X[i-1]]), np.asarray([X[i]]))
            if self.max_window_size == self.window.n_samples:
                self.estimator.partial_fit(self.window.get_attributes_matrix().reshape((1,-1)), 
                                           self.window.get_targets_matrix()[-1].reshape((1,-1)), sample_weight=sample_weight)
        self.first_fit = False
        return self

    def reset(self):
        self.window.reset()
        self.estimator.reset()
        
        return self
    
    def clone_window(self):
        window = InstanceWindow(n_features=self.window.n_attributes , n_targets=self.window.n_targets, 
                                categorical_list=self.window.categorical_attributes, max_size=self.window.max_size)
        window._buffer =  np.array(self.window._buffer)
        window._n_samples = self.window._n_samples
        
        return window

    def predict(self, X):
        """ Predicts the next value For all values in X.
        The estimator consider X[0] as the value conming exactly after the last partially fit value.
        
        Parameters
        ----------
        X: Numpy.ndarray of shape (n_samples, n_features)
            All the samples we want to predict the label for.
            
        Returns
        -------
        list
            A list containing the predicted values for all instances in X.
        
        """
        
        if len(X.shape) == 1:
            X = X.reshape(-1,1)
        r = X.shape[0]
        
        window = self.clone_window()
        
        predictions = []
        for i in range(r):
            window.add_element(np.asarray([X[i]]), np.asarray([X[i]]))
            if self.max_window_size == self.window.n_samples:
                pred = self.estimator.predict(window.get_attributes_matrix().reshape((1,-1)))
                if(len(pred.flatten()) == 1):
                    pred = pred[0]
                predictions.append(pred[0])
            
        return np.array(predictions)
    
    def predict_proba(self, X):
        """
        Method not implemented for this Estimator
        """
        raise NotImplementedError
    
    def forcast(self, X, n_steps):
        """ Predicts the next n_steps values coming after all values in X.
        The estimator consider X[0] as the value conming exactly after the last partially fit value.
        
        Parameters
        ----------
        X: Numpy.ndarray of shape (n_samples, n_features)
            All the samples we want to predict the next value for.
        n_steps: 
            The number of values to Forcast
        Returns
        -------
        list
            A list containing the predicted n_steps to come after values in X.
         
        """
        if len(X.shape) == 1:
            X = X.reshape(-1,1)
        r = X.shape[0]
        
        window = self.clone_window()
        
        for i in range(r):
            window.add_element(np.asarray([X[i]]), np.asarray([X[i]]))
        
        forecasts = []
        for i in range(n_steps):
            next_element = self.estimator.predict( window.get_attributes_matrix().reshape((1,-1)))
            window.add_element(next_element.reshape((1,-1)), next_element.reshape((1,-1)))
            if(len(next_element.flatten()) == 1):
                next_element = next_element[0]
            forecasts.append(next_element[0])
            
        return np.asarray(forecasts)