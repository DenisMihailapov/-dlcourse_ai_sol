import numpy as np


def softmax(predictions):
    '''
    Computes probabilities from scores

    Arguments:
      predictions, np array, shape is either (N) or (batch_size, N) -
        classifier output

    Returns:
      probs, np array of the same shape as predictions - 
        probability for every class, 0..1
    '''
    # TODO implement softmax
    exp = np.exp(predictions - np.max(predictions))
    
    # probs for (N) shape predictions
    if predictions.shape == (len(predictions), ):
        probs = exp / np.sum(exp)
        
    # probs for (batch_size, N) predictions
    else:
        probs = exp / np.sum(exp, axis=1, keepdims=True)
    
    return probs


def cross_entropy_loss(probs, target_index):
    '''
    Computes cross-entropy loss

    Arguments:
      probs, np array, shape is either (N) or (batch_size, N) -
        probabilities for every class
      target_index: np array of int, shape is (1) or (batch_size) -
        index of the true class for given sample(s)

    Returns:
      loss: single value
    '''
    # TODO implement cross-entropy
    
    # loss for (N) shape probs
    if probs.shape == (len(probs), ):
      loss = - np.log(probs[target_index])
        
    # loss for (batch_size, N) shape probs
    else:
      loss = - np.log(probs[np.arange(len(probs)), target_index])
      #use matrix coordinates where 
      #    colums from target_index,
      #    rows   from np.arange(len(probs))
    
    return loss


def softmax_with_cross_entropy(predictions, target_index):
    '''
    Computes softmax and cross-entropy loss for model predictions,
    including the gradient

    Arguments:
      predictions, np array, shape is either (N) or (batch_size, N) -
        classifier output
      target_index: np array of int, shape is (1) or (batch_size) -
        index of the true class for given sample(s)

    Returns:
      loss, single value - cross-entropy loss
      dprediction, np array same shape as predictions - gradient of predictions by loss value
    '''
    # TODO implement softmax with cross-entropy
    
    S = softmax(predictions)
    loss = cross_entropy_loss(S, target_index).mean()

    
    mask = np.zeros_like(predictions)

    # mask and dprediction for (N) shape predictions
    if predictions.shape == (len(predictions), ):
        mask[target_index] = 1
        dprediction = S - mask
        
    # mask and dprediction for (batch_size, N) shape predictions
    else:
        mask[np.arange(len(mask)), target_index] = 1
        dprediction = (S - mask) / (len(mask)) #batch norm

    return loss, dprediction
    

def linear_softmax(X, W, target_index):
    '''
    Performs linear classification and returns loss and gradient over W

    Arguments:
      X, np array, shape (num_batch, num_features) - batch of images
      W, np array, shape (num_features, classes) - weights
      target_index, np array, shape (num_batch) - index of target classes

    Returns:
      loss, single value - cross-entropy loss
      dW, np.array same shape as W - gradient of weight by loss

    '''
    predictions = np.dot(X, W)

    # TODO implement prediction and gradient over W
    loss, dprediction = softmax_with_cross_entropy(predictions, target_index)
    dW = np.dot(X.T, dprediction)
    
    return loss, dW


def l2_regularization(W, reg_strength):
    '''
    Computes L2 regularization loss on weights and its gradient

    Arguments:
      W, np array - weights
      reg_strength - float value

    Returns:
      loss, single value - l2 regularization loss
      gradient, np.array same shape as W - gradient of weight by l2 loss
    '''

    # TODO: implement l2 regularization and gradient
    loss = reg_strength * np.sum(W ** 2)
    grad = 2 * reg_strength * W

    return loss, grad


class LinearSoftmaxClassifier():
    def __init__(self, learning_rate=1e-7, reg_rate=1e-5 ):

      self.learn_rate = learning_rate
      self.reg_rate   = reg_rate

      self.W, self.dW, self.l2_dW = None, None, None

      self.loss_history = []
    
    def get_random_batches(self, X, y, batch_size):
      batches_indices = \
          np.array_split(\
              np.arange(X.shape[0]), \
              np.arange(batch_size, X.shape[0], batch_size))
      batch_ind = batches_indices[np.random.randint(len(batches_indices))]

      return X[batch_ind], y[batch_ind]

    def grad_and_loss(self, X_batch, y_batch):

      loss,    self.dW    = linear_softmax(X_batch, self.W, y_batch)
      l2_loss, self.l2_dW = l2_regularization(self.W, self.reg_rate)

      self.loss_history.append(loss + l2_loss)

    def step(self, learn_rate = None):
      self.W -= (self.learn_rate if learn_rate is None else learn_rate)\
              * (self.dW + self.l2_dW)

    def fit(self, X, y, epochs=1, batch_size=100, learning_rate=None, reg_rate=None, verbose=True):
        
      if learning_rate:self.learn_rate = learning_rate
      if reg_rate     :self.reg_rate   = reg_rate

      num_features = X.shape[1]
      num_classes  = np.max(y)+1

      if  self.W is None:
          self.W = 0.001 * np.random.randn(num_features, num_classes)


      W = self.W.copy()
      self.loss_history = []
      X_batch, y_batch = None, None      
      for epoch in range(epochs):
        
        X_batch, y_batch = self.get_random_batches(X, y, batch_size)

        self.grad_and_loss(X_batch, y_batch)
            
        self.step()

        # end
        if verbose and epoch % 10 == 0:
          print("Epoch %i, loss: %f" % (epoch, self.loss_history[-1]))
                
      self.W = W

      return self.loss_history

    def predict(self, X):
        '''
        Produces classifier predictions on the set
       
        Arguments:
          X, np array (test_samples, num_features)

        Returns:
          y_pred, np.array of int (test_samples)
        '''

        return np.argmax(np.dot(X, self.W), axis=1)



                
                                                          

            

                
