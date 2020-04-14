# -*- coding: utf-8 -*-
"""
@author: lgonon
"""

## Solve FBSDE for the equilibrium problem with exogenous volatility and create
## Figure 4 in the paper.

## In the first part of the code (Sections 0., 1. and 2.) all necessary objects
## and functions are introduced. In the second part (Section 3.) these are then
## called appropriately to build, train and evaluate the neural networks used
## for solving the FBSDE in the paper.


##############################################################################
######################################### 0. Import packages #################
##############################################################################


import time
import tensorflow as tf # Code was written for Tensorflow 1, but still runs with warnings under 2.
import numpy as np
from scipy.stats import multivariate_normal as normal
import os
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp


##############################################################################
######################################### 1. Specify cost structure ##########
##############################################################################


class QuadraticCost(object):
  def __init__(self, T,sigma, beta, gamma):
    self.learning_rate = 0.01
    self.n_maxstep = 2000
    self.gamma = gamma
    self.sigma = sigma
    self.T = T
    self.c1 = np.sum(self.gamma)*self.sigma**2
    self.lam = 0.1
    self.sqrtc = np.sqrt(0.5*self.c1*self.lam)
    self.q = 2.
    self.save = False

  def f_tf(self, t, X, Y, Z):
      return -self.c1*0.5*X
  def x_drift(self, Y):
      return Y
  def x_drift_np(self,Y):
      return Y
  def benchmark(self,t,x):
      return 1./self.lam*self.sqrtc*np.tanh(self.sqrtc*(t-self.T))*x
  def long_run_benchmark(self,x):
      return -1./self.lam*np.sqrt(np.sum(self.gamma)*0.5*self.lam)*self.sigma*x
      


class PowerCost(object):
  def __init__(self, T,sigma, beta, gamma):
    self.learning_rate = 0.001#0.0005
    self.n_maxstep = 50000
    self.gamma = gamma
    self.sigma = sigma
    self.T = T
    self.c1 = np.sum(self.gamma)*self.sigma**2
    self.sqrtc = np.sqrt(0.5*self.c1)
    self.q = 1.5
    self.lam = 5.22e-6
    self.save=True

  def f_tf(self, scale,t, X, Y, Z):
      return (-self.c1*0.5*scale)*X
  def x_drift(self, scale,Y):
      return Y*tf.abs(Y)*(scale*1./np.power(self.lam,1./(self.q-1.)))#tf.sign(Y)*tf.pow(tf.abs(Y),1./(self.q-1.))*(1./s*1./np.power(self.lam,1./(self.q-1.)))
  def x_drift_np(self,Y):
      return np.sign(Y)*np.power(np.abs(Y),1./(self.q-1.))*1./np.power(self.lam,1./(self.q-1.))


##############################################################################
######################################### 2. FBSDE Solver ####################
##############################################################################


class FBSDESolver(object):
    """Solve FBSDE for Equilibrium."""
    def __init__(self, sess,ex_dict=None):
        ## Model and parameter specifications, these can also be changed by 
        ## optional dictionary input (ex_dict) (see below) 

        self.d = 1
        self.n_time = 25
        self.T = 25
        self.T_true = 25
        self.T_ratio = self.T_true/self.T
        self.s_outstanding = 2.46e11
        gbar = 8.31e-14
        kappa = 0.5
        self.sigma = 1.88
        self.beta = np.array([2.33e10,-2.33e10])
        self.gamma = np.array([gbar*(kappa+1.),gbar*(kappa+1.)/kappa])
        self.phiini = self.s_outstanding*self.gamma[1]/np.sum(self.gamma)
        self.model = PowerCost(self.T,self.sigma,self.beta,self.gamma)
        ## Scaling
        self.x_scale = 1./self.s_outstanding
        self.y_scale = 1./(np.power(self.s_outstanding,3./4.)*np.sqrt(self.model.lam*np.sum(self.gamma)/2.)*self.sigma*self.T_ratio)
        ## Time grid and horizon
        self.NSim = self.n_time
        self.dtSim = (self.T+0.0)/self.NSim
        self.tSim = np.arange(0, self.NSim+1)*self.dtSim
        self.sqrt_dtSim = np.sqrt(self.dtSim)
        ## Further variables
        self.sess = sess
        self.Yini = [-0.01, 0.01]
        ## Network architecture
        self.D = self.d
        self.n_neuron = [self.D, self.D*15, self.D*15, self.D]
        self.n_layer = 4
        ## Training         
        self.batch_size = 128
        self.valid_size = 4*256
        self.learning_rate = self.model.learning_rate
        self.load_parameters = False
        self.n_maxstep = self.model.n_maxstep
        self.save_dir = os.getcwd() + '\equlibrium.ckpt'

        ### Further constants and variables for training
        self.n_displaystep = 100
        
        ### Here any of the above are replaced by values in ex_dict
        if ex_dict != None:
            for param_name,param_value in ex_dict.items():
                setattr(self,param_name,param_value)
         

    def train(self):
        # Train the network or restore previously trained parameters 
        start_time = time.time()
        # train operations
        self.global_step = tf.get_variable('global_step', [],initializer=tf.constant_initializer(1),trainable=False, dtype=tf.int32)
        trainable_vars = tf.trainable_variables()
        grads = tf.gradients(self.loss, trainable_vars)
        optimizer = tf.train.AdamOptimizer(self.learning_rate)
        apply_op = optimizer.apply_gradients(list(zip(grads, trainable_vars)),global_step=self.global_step)
        train_ops = [apply_op]
        self.train_op = tf.group(*train_ops)
        self.loss_history = []
        self.init_history = []
        # for validation
        time_temp = time.time()-start_time
        dW_valid  = self.sample_path(self.valid_size)
        start_time = time.time()
        feed_dict_valid = {self.dW: dW_valid, self.is_training: False, self.is_evaluating: False, self.Xfull: dW_valid}
        # initialization
        step = 1
        self.sess.run(tf.global_variables_initializer())
        temp_loss = self.sess.run(self.loss,feed_dict=feed_dict_valid)
        temp_init = self.Y0.eval(self.sess)[0]
        self.loss_history.append(temp_loss)
        self.init_history.append(temp_init)
        print("step: %5u,  loss: %.4e,  " % (0, temp_loss) + "Y0: %.4e,  time: %4u" % (temp_init, time.time()-start_time+self.t_bd))
        # possibly restore previously trained variables:
        if self.load_parameters:
            saver = tf.train.Saver(trainable_vars)
            saver.restore(self.sess,self.save_dir)
            print("Restored trained parameters from: %s" % self.save_dir)   
        # begin sgd iteration
        for _ in range(self.n_maxstep+1):
            step = self.sess.run(self.global_step)
            time_temp = time_temp + time.time()-start_time
            dW_train = self.sample_path(self.batch_size)
            start_time = time.time()
            self.sess.run(self.train_op,feed_dict={self.dW: dW_train, self.is_training: True,self.is_evaluating: False, self.Xfull: dW_valid})
            if step % self.n_displaystep == 0:
                temp_loss = self.sess.run(self.loss,feed_dict=feed_dict_valid)
                temp_init = self.Y0.eval(self.sess)[0]
                self.loss_history.append(temp_loss)
                self.init_history.append(temp_init)
                print("step: %5u,  loss: %.4e,  " % (step, temp_loss) + "Y0: %.4e,  time: %4u" % (temp_init, time_temp+ time.time()-start_time))  
            step += 1
        self.total_time = time.time() - start_time + time_temp +self.t_bd
        print("running time: %.3f s" % (self.total_time))
    
    def build(self):
        # Construct the network for training or analysis.
        start_time = time.time()
        # Build the whole network by stacking subnetworks
        self.dW = tf.placeholder(tf.float64,[None, self.d, self.n_time],name='dW')
        self.is_training = tf.placeholder(tf.bool)
        self.Y0 = tf.Variable(tf.random_uniform([1],minval=self.Yini[0],maxval=self.Yini[1],dtype=tf.float64))
        self.Z0 = tf.Variable(tf.random_uniform([1, self.d],minval=-.1,maxval=.1,dtype=tf.float64))
        self.allones = tf.ones(shape=tf.stack([tf.shape(self.dW)[0], 1]),dtype=tf.float64)
        self.is_evaluating = tf.placeholder(tf.bool)
        Y = self.allones * self.Y0
        Z = tf.matmul(self.allones, self.Z0)
        self.Xfull = tf.placeholder(tf.float64,[None, self.d, self.n_time],name='Xfull')
        X = tf.cond(self.is_evaluating,lambda: self.Xfull[:,:,0], lambda: tf.constant(0.,tf.float64)*Y + tf.constant(self.phiini*self.x_scale-self.gamma[1]*self.s_outstanding*self.x_scale/np.sum(self.gamma),tf.float64)*self.allones)
        self.Z_history = Z[:,:,tf.newaxis]
        self.Y_history = Y[:,:,tf.newaxis]
        self.X_history = X[:,:,tf.newaxis]
        with tf.variable_scope('forward') as scope:
            for t in range(0, self.n_time-1):
                Yincr = tf.reduce_sum(Z*self.dW[:, :, t], 1, keep_dims=True) - self.model.f_tf(self.T_ratio**2*self.y_scale/self.x_scale,self.tSim[t],
                                  X, Y, Z)*self.dtSim 
                X = tf.cond(self.is_evaluating,lambda: self.Xfull[:,:,t+1],
                            lambda: X + self.model.x_drift(self.x_scale/self.y_scale**2,Y)*self.dtSim + self.dW[:,:,t]*(self.x_scale*self.beta[0]/self.sigma)*1./np.sqrt(self.T_ratio))

                Z = self.FFN(X,str(t+1))/self.d
                Y = Y + Yincr
                self.Z_history = tf.concat([self.Z_history,Z[:,:,tf.newaxis]],axis=2)
                self.Y_history = tf.concat([self.Y_history,Y[:,:,tf.newaxis]],axis=2)
                self.X_history = tf.concat([self.X_history,X[:,:,tf.newaxis]],axis=2)
            # terminal time
            Y = Y - self.model.f_tf(self.T_ratio**2*self.y_scale/self.x_scale,self.tSim[self.n_time-1],
                              X, Y, Z)*self.dtSim
            Y = Y + tf.reduce_sum(Z*self.dW[:, :, self.n_time-1], 1,
                                  keep_dims=True)
            # not needed, but in principle also have to update this:
            #X = X + self.model.x_drift(Y)*self.dtSim+ self.dW[:,:,self.n_time-1]*self.beta[0]/self.sigma
            self.Y_history = tf.concat([self.Y_history,Y[:,:,tf.newaxis]],axis=2)
            term_delta = Y
            self.clipped_delta = \
                tf.clip_by_value(term_delta, -5000.0, 5000.0)
            self.loss = tf.reduce_mean(self.clipped_delta**2)
        self.t_bd = time.time()-start_time 
    

    def sample_path(self, n_sample):
        dW_sample = np.zeros([n_sample, self.d, self.n_time])
        for i in range(self.n_time):
            dW_sample[:, :, i] = np.reshape(normal.rvs(mean=np.zeros(self.d),cov=1, size=n_sample)*self.sqrt_dtSim,(n_sample, self.d))
        return dW_sample
    
    
    def FFN(self, x, name):
        with tf.variable_scope(name):
            
            x_norm = tf.concat([self._batch_norm(x[:,0:-1], name='layer0_normal'),x[:,-1,tf.newaxis]],axis=1)

            layer1 = self.FNN_layer(x_norm, self.n_neuron[1],
                                     name='layer1')
            layer2 = self.FNN_layer(layer1, self.n_neuron[2],
                                     name='layer2')
            z = self.FNN_layer(layer2, self.n_neuron[3],
                                activation_fn=None, name='final')
        return z

    def FNN_layer(self, input_, out_sz, 
                   activation_fn=tf.nn.relu,
                   std=5.0, name='linear'):
        with tf.variable_scope(name):
            shape = input_.get_shape().as_list()
            w = tf.get_variable('Matrix',
                                [shape[1], out_sz], tf.float64,
                                tf.random_normal_initializer(stddev= \
                                    std/np.sqrt(shape[1]+out_sz)))
            hidden = tf.matmul(input_, w)
            hidden_bn = self._batch_norm(hidden, name='normal')
        if activation_fn != None:
            return activation_fn(hidden_bn)
        else:
            return hidden_bn

    def _batch_norm(self, x, name):
        """Batch normalization"""
        with tf.variable_scope(name):
            params_shape = [x.get_shape()[-1]]
            beta = tf.get_variable('beta', params_shape,
                                   tf.float64,
                                   tf.random_normal_initializer(0.0, stddev=0.1,
                                             dtype=tf.float64))
            gamma = tf.get_variable('gamma', params_shape,
                                    tf.float64,
                                    tf.random_uniform_initializer(0.1, 0.5,
                                              dtype=tf.float64))    
            mean, variance = tf.nn.moments(x, [0], name='moments')
            y = tf.nn.batch_normalization(x, mean, variance, 
                                          beta, gamma, 1e-6)
            y.set_shape(x.get_shape())
            return y

##############################################################################
######################################### 3. Run FBSDE Solver ################
##############################################################################
          


    
if __name__ == '__main__':
    do_training = False #set to False to load a previously trained network
    create_csv = False # set to True to create csv files with the data    
    np.random.seed(1)
    #### 3.1. Solve the FBSDE ####
    
    ## Build and train/load the FBSDE solver
    tf.reset_default_graph()
    sess=tf.Session()
    tf.set_random_seed(1)
    print("Begin to find Equilibrium")
    if do_training:
        Equilibrium = FBSDESolver(sess)
        Equilibrium.build()
        Equilibrium.train() 
        if Equilibrium.model.save:
            saver = tf.train.Saver(tf.trainable_variables())
            save_path = saver.save(sess,Equilibrium.save_dir)
            print("Model saved in file: %s" % save_path)  
    else:
        ex_dict={'load_parameters':True,'n_maxstep':1,'learning_rate':1e-8}
        Equilibrium = FBSDESolver(sess,ex_dict)        
        Equilibrium.build()
        Equilibrium.train()    
        
    ## Generate sample paths for testing   
    dW_test = Equilibrium.sample_path(100000)
    
    feed_dict_test = {Equilibrium.dW: dW_test, Equilibrium.is_training: False, Equilibrium.is_evaluating: False, Equilibrium.Xfull: dW_test}
    print(Equilibrium.sess.run(Equilibrium.loss,feed_dict=feed_dict_test))
    ## Evaluate FBSDE solution on test paths
    Z_sample = Equilibrium.sess.run(Equilibrium.Z_history,feed_dict=feed_dict_test)
    X_sample = Equilibrium.sess.run(Equilibrium.X_history,feed_dict=feed_dict_test)
    Y_sample = Equilibrium.sess.run(Equilibrium.Y_history,feed_dict=feed_dict_test)
    
    
    #### 3.2 Solve the benchmark ODE ####
    
    ## Set parameters
    q = Equilibrium.model.q
    be = Equilibrium.beta[0]/Equilibrium.sigma
    gam = np.sum(Equilibrium.model.gamma)*0.5
    sigma = Equilibrium.sigma
    lam = Equilibrium.model.lam
    
    ## Define ODE RHS and boundary conditions and solve ODE
    ## Note: we solve the normalized 1st order ode and also display the value 
    ## at 0 to verify that the constant c is chosen correctly
    c = 1.7714393 
    def scale_ode(t, y): return -np.power(q,(-q/(q-1.)))*(q-1.)*np.power(np.abs(y),q/(q-1.))+np.power(t,2)-c    
    def ggrowth(x): return np.power((1./(q-1.)*(x**2)),(q-1.)/q)*q
    xnegInfty = -100.
    gnegInfty = ggrowth(xnegInfty)    
    sol = solve_ivp(scale_ode, [xnegInfty,0],[gnegInfty], dense_output=True)
    print(sol.y[0,-1])
    ## Rescale to get ODE with required parameters
    delta = be
    c0 = np.power(q,-3./(q+2.))*np.power(delta,4.*(q-1)/(q+2.))*np.power(lam,3./(q+2.))*np.power(gam/8.*sigma**2,(q-1.)/(q+2.))
    c1 = np.power(2.,(q-1.)/(q+2.))*np.power(delta,-2.*q/(q+2.))*np.power(lam,-1./(q+2.))*np.power(q*gam*sigma**2,1./(q+2.))
    def g(x): return c0*sol.sol(x*c1)
        
    #### 3.3 Create Figure 4 in the paper ####

    def H_fun(x): return np.sign(x)*np.power(np.abs(x),1./(q-1.))*1./np.power(lam,1./(q-1.))    
    X_sample = X_sample/Equilibrium.x_scale
    Y_sample = Y_sample / Equilibrium.y_scale
    xmin = np.percentile(X_sample,1)
    xmax = np.percentile(X_sample,99)
    ymin = Equilibrium.model.x_drift_np(np.percentile(Y_sample,0.1))
    ymax = Equilibrium.model.x_drift_np(np.percentile(Y_sample,99.99))
    for i in range(1,25):
        ind = np.argsort(X_sample[:,0,i],axis=0)
        indcrit = (X_sample[:,0,i][ind]>=0).argmax(axis=0)
        solneg = g(X_sample[:,0,i][ind][0:indcrit].copy())
        solpos = g(-np.flip(X_sample[:,0,i][ind][indcrit:].copy(),axis=0))
        
        soly = np.concatenate((solneg[0,:],-np.flip(solpos[0,:],axis=0)),axis=0)
        plt.figure()
        Nactual = soly.shape[0]
        lowery = np.percentile(Equilibrium.model.x_drift_np(Y_sample[:,:,i])[ind],0.1)
        uppery = np.percentile(Equilibrium.model.x_drift_np(Y_sample[:,:,i])[ind],99.9)
        ind2 = (Equilibrium.model.x_drift_np(Y_sample[:,:,i])[ind]>lowery)*(Equilibrium.model.x_drift_np(Y_sample[:,:,i])[ind]<uppery)
        ind2 = ind2[:,0]
        plt.plot(X_sample[:,0,i][ind][ind2],Equilibrium.model.x_drift_np(Y_sample[:,:,i])[ind][ind2],label='Neural Network BSDE')
        plt.plot(X_sample[:,0,i][ind][:Nactual],H_fun(soly),'r--',label='Long-run ODE')
        plt.legend()
        axes = plt.gca()
        axes.set_xlim([xmin,xmax])
        axes.set_ylim([ymin,ymax])
        plt.xlabel('State variable at $t$')
        plt.ylabel('Optimal trading rate at $t$')
        if i==2: #or i==24:
            plt.savefig('ODEbenchmark'+str(i)+'.pdf') 
            if create_csv:
                np.savetxt('Figure4'+str(i)+'BSDEX.csv',X_sample[:,0,i][ind][ind2],delimiter=',')
                np.savetxt('Figure4'+str(i)+'BSDEY.csv',Equilibrium.model.x_drift_np(Y_sample[:,:,i])[ind][ind2],delimiter=',')
                np.savetxt('Figure4'+str(i)+'ODEX.csv',X_sample[:,0,i][ind][:Nactual],delimiter=',')
                np.savetxt('Figure4'+str(i)+'ODEY.csv',H_fun(soly),delimiter=',')
                