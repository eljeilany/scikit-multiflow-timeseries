from skmultiflow.core import BaseSKMObject, ClassifierMixin
from skmultiflow.utils.data_structures import InstanceWindow
from skmultiflow.utils.utils import *

import warnings

class TimeSeriesClassifier(BaseSKMObject, ClassifierMixin):
    
    def __init__(self, estimator, max_window_size=100):
        super().__init__()
        
        if not isinstance(estimator, ClassifierMixin):
            raise ValueError("estimator must be a Classifier, "
                                     "Call TimeSeriesClassifier with an instance of ClassifierMixin")
        
        self.max_window_size = max_window_size
        self.estimator = estimator
        self.window = InstanceWindow(max_size=max_window_size, dtype=float)
        self.first_fit = True
    
    def partial_fit(self, X, y=None, classes = None, sample_weight=None):
        """ Partially fits the model on the samples X and corresponding targets y.
        If Y is not provided the model will take X[t+1] as target for X[t] 
        
        Parameters
        ----------
        X: Numpy.ndarray of shape (n_samples, n_features)
            The data upon which the algorithm will create its model. 
            If y is not provided the value X[t+1] in X will be used as target For X[t]
            
        y: numpy.ndarray, optional
            An array-like containing the targets for all samples in X.
            y must have the shape as X
            
        classes: numpy.ndarray, optional (default=None)
            Array with all possible/known classes.
        sample_weight: Not used.
        
        Returns
        -------
        TimeSeriesClassifier
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
                                           self.window.get_targets_matrix()[-1].reshape((1,)), classes = classes, sample_weight=sample_weight)
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
        """ Predicts the next class label coming after all values in X.
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
        
        proba = self.predict_proba(X)
        predictions = []
        for i in range(r):
            predictions.append(np.argmax(proba[i]))
        return np.array(predictions)

    def predict_proba(self, X):
        """ Estimates the probability of each sample in X[t+1] belonging to each of the class-labels.
        
        Parameters
        ----------
        X: Numpy.ndarray of shape (n_samples, n_features)
        
        Returns
        -------
        numpy.ndarray
            An array of shape (n_samples, n_features), in which each outer entry is 
            associated with the X entry of the same index. And where the list in 
            index [i] contains len(self.target_value) elements, each of which represents
            the probability that the i-th sample of X belongs to a certain label.
        """
        if len(X.shape) == 1:
            X = X.reshape(-1,1)
        r = X.shape[0]
        
        window = self.clone_window()
        
        proba = []
        for i in range(r):
            window.add_element(np.asarray([X[i]]), np.asarray([X[i]]))
            if self.max_window_size == self.window.n_samples:
                pred = self.estimator.predict_proba(window.get_attributes_matrix().reshape((1,-1)))
                proba.append(pred)

        return np.asarray(proba)
    
    def forcast(self, X, n_steps):
        """ Predicts the next n_steps class labels coming after all values in X.
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
            next_element = self.estimator.predict_proba( window.get_attributes_matrix().reshape((1,-1)))
            window.add_element(np.argmax(next_element, axis=-1).reshape((1,-1)), np.argmax(next_element, axis=-1).reshape((1,-1)))
            forecasts.append(np.argmax(next_element, axis=-1)[0])
            
        return np.asarray(forecasts)