import os
os.environ["THEANO_FLAGS"] = "device=gpu"
from sklearn.base import BaseEstimator
import os
from lasagne import layers, nonlinearities, objectives, updates, init
from lasagne.updates import nesterov_momentum
from nolearn.lasagne import NeuralNet, BatchIterator
import numpy as np
#from nolearn.lasagne.base import objective
from lasagne.objectives import aggregate
#from lasagne.regularization import regularize_layer_params, l2, l1
from nolearn.lasagne.handlers import EarlyStopping
from skimage import data
from skimage import transform

lambda_regularization = 0.01

class FlipBatchIterator(BatchIterator):
    def transform(self, Xb, yb):
        Xb, yb = super(FlipBatchIterator, self).transform(Xb, yb)
        # Flip half of the images in this batch at random:
        bs = Xb.shape[0]
        indices = np.random.choice(bs, bs / 2, replace=False)
        #Xb[indices] = Xb[indices, :, ::-1, :]
        X_tmp1 = Xb[indices, :, ::-1, :]
        Y_tmp1 = yb[indices]    
        indices = np.random.choice(bs, bs / 2, replace=False)
        #Xb[indices] = Xb[indices, :, :, ::-1]
        X_tmp2 = Xb[indices, :, :, ::-1]
        Y_tmp2 = yb[indices]    
        indices = np.random.choice(bs, bs / 2, replace=False)
        X_tmp3 = Xb[indices, :, :, :]
        Y_tmp3 = yb[indices]    
        X_tmp3 = X_tmp3.transpose((0,1,3,2)) 
        indices = np.random.choice(bs, bs / 2, replace=False)
        X_tmp4 = Xb[indices, :, :, ::-1]
        Y_tmp4 = yb[indices]    
        X_tmp4 = X_tmp3.transpose((0,1,3,2)) 
        indices = np.random.choice(bs, bs / 2, replace=False)
        X_tmp5 = Xb[indices, :, ::-1, :]
        Y_tmp5 = yb[indices]    
        X_tmp5 = X_tmp3.transpose((0,1,3,2))
        
        Xb = np.append(Xb,X_tmp1,axis=0)
        Xb = np.append(Xb,X_tmp2,axis=0)
        Xb = np.append(Xb,X_tmp3,axis=0)
        Xb = np.append(Xb,X_tmp4,axis=0)
        Xb = np.append(Xb,X_tmp5,axis=0)
        yb = np.append(yb,Y_tmp1)
        yb = np.append(yb,Y_tmp2)
        yb = np.append(yb,Y_tmp3)
        yb = np.append(yb,Y_tmp4)
        yb = np.append(yb,Y_tmp5)
        
        # small rotation of the images
        lx = 44
        pad_lx = 64
        shift_x = lx/2.
        shift_y = lx/2.
        tf_rotate = transform.SimilarityTransform(rotation=np.deg2rad(15))
        tf_shift = transform.SimilarityTransform(translation=[-shift_x, -shift_y])
        tf_shift_inv = transform.SimilarityTransform(translation=[shift_x, shift_y])
        
        indices = np.random.choice(bs, bs / 2, replace=False)
        X_tmp6 = Xb[indices, :, ::-1, :]
        X_tmp6 = X_tmp6.transpose(0,2,3,1)
        X_tmp6 = np.pad(X_tmp6,((0,0),(10,10),(10,10),(0,0)),'constant', constant_values=(0,0))
        Y_tmp6 = yb[indices]
        x_rot = X_tmp6[0]
        x_rot = x_rot.reshape(1,pad_lx,pad_lx,3)
        for i in X_tmp6[1::]:
            xdel = transform.warp(i, (tf_shift + (tf_rotate + tf_shift_inv)).inverse)            
            xdel=xdel.reshape(1,pad_lx,pad_lx,3)
            x_rot=np.append(x_rot,xdel,axis=0)
        
        x_rot = x_rot[:, 10:54, 10:54, :]
        x_rot = x_rot.transpose(0,3,1,2)
        x_rot = x_rot.astype(np.float32)
        Xb = np.append(Xb,x_rot,axis=0)
        yb = np.append(yb,Y_tmp6)
        return Xb, yb


def build_model(hyper_parameters):
    net = NeuralNet(
        layers=[
            ('input', layers.InputLayer),
            ('conv1', layers.Conv2DLayer),
            ('pool1', layers.MaxPool2DLayer),
            ('conv2', layers.Conv2DLayer),
            ('pool2', layers.MaxPool2DLayer),
            ('conv3', layers.Conv2DLayer),
            ('pool3', layers.MaxPool2DLayer),
            ('hidden4', layers.DenseLayer),
            #('dropout4', layers.DropoutLayer),
            ('hidden5', layers.DenseLayer),
            ('dropout5', layers.DropoutLayer),
            ('output', layers.DenseLayer),
            ],
        input_shape=(None, 3, 44, 44),
        use_label_encoder=True,
        # objective function
        # objective=objective_with_L2,
        verbose=1,
        **hyper_parameters
        )
    return net

hyper_parameters = dict(
    conv1_num_filters=64, conv1_filter_size=(4, 4), pool1_pool_size=(2, 2),
    conv2_num_filters=128, conv2_filter_size=(4, 4), pool2_pool_size=(2, 2),
    conv3_num_filters=128, conv3_filter_size=(3, 3), pool3_pool_size=(2, 2),
    hidden4_num_units=1000, hidden4_nonlinearity = nonlinearities.leaky_rectify,
    #hidden4_regularization = lasagne.regularization.l2(hidden4),
    hidden5_num_units=1000, hidden5_nonlinearity = nonlinearities.leaky_rectify,
    dropout5_p=0.3,
    #hidden5_regularization = regularization.l2,
    output_num_units=18, 
    output_nonlinearity=nonlinearities.softmax,
    update_learning_rate=0.01,
    #update_momentum=0.9,
    update=updates.adagrad,
    max_epochs=150,
    
    # handlers
    on_epoch_finished = [EarlyStopping(patience=40, criterion='valid_accuracy', criterion_smaller_is_better=False)],
    batch_iterator_train=FlipBatchIterator(batch_size=150)
)


class Classifier(BaseEstimator):

    def __init__(self):
        self.net = build_model(hyper_parameters)

    def preprocess(self, X):
        X = (X / 255.)
        X = X.astype(np.float32)
        X = X[:, 10:54, 10:54, :]
        X = X.transpose((0, 3, 1, 2))
        return X
    
    def preprocess_y(self, y):
        return y.astype(np.int32)

    def fit(self, X, y):
        X = self.preprocess(X)
        
        
        self.net.fit(X, self.preprocess_y(y))
        return self

    def predict(self, X):
        X = self.preprocess(X)
        return self.net.predict(X)

    def predict_proba(self, X):
        X = self.preprocess(X)
        return self.net.predict_proba(X)
