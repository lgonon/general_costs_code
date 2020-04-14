# -*- coding: utf-8 -*-
"""
@author: lgonon
"""
## Solve FBSDE for the equilibrium problem with endogenous volatility and 
## create Figures 5 and 6 in the paper as well as additional plots.

## In the first part of the code (Sections 0.-3.) all necessary objects
## and functions are introduced. In the second part (Section 4.) these are then
## called appropriately to build, train and evaluate the neural networks used
## for solving the FBSDE in the paper.


##############################################################################
######################################### 0. Import packages #################
##############################################################################

import time
import tensorflow as tf #Code was written under v1, but still runs under v2 
import numpy as np
from scipy.stats import multivariate_normal as normal
import os
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import scipy.interpolate


##############################################################################
######################################### 1. Specify cost structure ##########
##############################################################################


class QuadraticCost(object):
  def __init__(self, T, gamma):
    self.learning_rate = 0.0005#0.0025
    self.n_maxstep = 70000
    self.gamma = gamma
    self.T = T
    self.lam = 1.08e-10
    self.save = True
    self.stradd = ''
    self.x_scale = self.lam
    self.beta = np.array([2.19e10,-2.19e10])

  def x_drift(self, Y):
      return Y
  def x_drift_np(self,Y):
      return Y
  
class PowerCost(object):
  def __init__(self, T, gamma):
    self.learning_rate = 0.0005
    self.n_maxstep = 80000#50000
    self.gamma = gamma
    self.T = T
    self.lam = 5.22e-6
    self.q = 1.5
    self.save = True
    self.stradd = 'pow'
    self.x_scale = 1./2.46e11
    self.beta = np.array([2.33e10,-2.33e10])

  def x_drift(self, Y):
      return tf.sign(Y)*tf.pow(tf.abs(Y),1./(self.q-1.))*1./np.power(self.lam,1./(self.q-1.))
  def x_drift_np(self,Y):
      return np.sign(Y)*np.power(np.abs(Y),1./(self.q-1.))*1./np.power(self.lam,1./(self.q-1.))

##############################################################################
######################################### 2. FBSDE Solver ####################
##############################################################################


class FBSDESolver(object):
    """Solve FBSDE for Equilibrium."""
    def __init__(self, sess,ex_dict=None,ispower=False):
        ## Model and parameter specifications, these can also be changed by 
        ## optional dictionary input (ex_dict) (see below) 
        
        self.d = 1
        self.n_time = 40
        
        self.T = 20
        self.s_outstanding = 2.46e11
        gbar = 8.31e-14
        kappa = 0.5
        self.a = 1.88
        self.gamma = np.array([gbar*(kappa+1.),gbar*(kappa+1.)/kappa])        
        self.phiini = self.s_outstanding*self.gamma[1]/np.sum(self.gamma)
        self.s0 = 64.11
        self.b = self.s0/self.T+self.s_outstanding*self.a**2*self.gamma[0]*self.gamma[1]/np.sum(self.gamma)        
        #b=S0/T+s a^2 \bar{\gamma}
        
        self.ispower = ispower
        self.model = QuadraticCost(self.T,self.gamma)
        if ispower:
            self.model = PowerCost(self.T,self.gamma)
        ## Time grid and horizon
        self.NSim = self.n_time
        self.dtSim = (self.T+0.0)/self.NSim
        self.tSim = np.arange(0, self.NSim+1)*self.dtSim
        self.sqrt_dtSim = np.sqrt(self.dtSim)
        ## Further variables
        self.sess = sess
        self.Yini = [0.3, 0.6]
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
        self.save_dir = os.getcwd() + '\equlibriumStochVol' + self.model.stradd +'.ckpt'

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
        self.S0 = tf.Variable(tf.random_uniform([1, self.d],minval=-.1+self.s0,maxval=.1+self.s0,dtype=tf.float64))
        self.V0 = tf.Variable(tf.random_uniform([1, self.d],minval=-0.05+1.,maxval=.1+1.,dtype=tf.float64))
        self.allones = tf.ones(shape=tf.stack([tf.shape(self.dW)[0], 1]),dtype=tf.float64)
        self.is_evaluating = tf.placeholder(tf.bool)
        Y = self.allones * self.Y0
        Z = tf.matmul(self.allones, self.Z0)
        S = self.allones * self.S0
        V = tf.matmul(self.allones, self.V0)
        self.Xfull = tf.placeholder(tf.float64,[None, self.d, self.n_time],name='Xfull')
        X = tf.cond(self.is_evaluating,lambda: self.Xfull[:,:,0], lambda: tf.constant(0.,tf.float64)*Y + tf.constant(self.phiini,tf.float64)*self.allones*self.model.x_scale)
        self.Z_history = Z[:,:,tf.newaxis]
        self.Y_history = Y[:,:,tf.newaxis]
        self.X_history = X[:,:,tf.newaxis]
        self.S_history = S[:,:,tf.newaxis]
        self.V_history = V[:,:,tf.newaxis]
        W0 = tf.zeros(shape=tf.stack([tf.shape(self.dW)[0],1]),dtype=tf.float64)
        W = W0 #tf.cumsum(tf.concat([W0,self.dW],axis=2),axis=2)   
        with tf.variable_scope('forward') as scope:
            for t in range(0, self.n_time):
                if not self.ispower:
                    Yincr = tf.reduce_sum(Z*self.dW[:, :, t], 1, keep_dims=True) + (V*W*self.model.beta[0]*(self.a)-(V**2)*(self.a)**2*self.gamma[1]*self.s_outstanding/np.sum(self.gamma)+X*(self.a)**2*1./self.model.x_scale*(V**2))*np.sum(self.gamma)*0.5*self.model.x_scale*1./self.model.lam*self.dtSim 
                    Sincr = tf.reduce_sum(V*self.dW[:, :, t], 1, keep_dims=True) + (V*W*self.model.beta[0]*0.5*(self.gamma[0]-self.gamma[1])+(V**2)*self.gamma[1]*0.5*self.a*self.s_outstanding+X*self.a*1./self.model.x_scale*(V**2)*0.5*(self.gamma[0]-self.gamma[1]))*self.dtSim 
                    W = W + self.dW[:,:,t]
                    X =  X + self.model.x_drift(Y)*self.dtSim
                if self.ispower:
                    Yincr = tf.reduce_sum(Z*self.dW[:, :, t], 1, keep_dims=True) + (V*W*self.model.beta[0]*(self.a)-(V**2)*(self.a)**2*self.gamma[1]*self.s_outstanding/np.sum(self.gamma)+X*(self.a)**2*1./self.model.x_scale*(V**2))*np.sum(self.gamma)*0.5*self.dtSim 
                    Sincr = tf.reduce_sum(V*self.dW[:, :, t], 1, keep_dims=True) + (V*W*self.model.beta[0]*0.5*(self.gamma[0]-self.gamma[1])+(V**2)*self.gamma[1]*0.5*self.a*self.s_outstanding+X*self.a*1./self.model.x_scale*(V**2)*0.5*(self.gamma[0]-self.gamma[1]))*self.dtSim 
                    W = W + self.dW[:,:,t]
                    Y = tf.clip_by_value(Y, -5000.0, 5000.0)
                    X =  X + self.model.x_scale*self.model.x_drift(Y)*self.dtSim
                Y = Y + Yincr
                S = S + Sincr
                self.Y_history = tf.concat([self.Y_history,Y[:,:,tf.newaxis]],axis=2)
                self.S_history = tf.concat([self.S_history,S[:,:,tf.newaxis]],axis=2)
                self.X_history = tf.concat([self.X_history,X[:,:,tf.newaxis]],axis=2)
                if t<(self.n_time-1):
                    Z = self.FFN(tf.concat([X,W],axis=1),str(t+1))/self.d
                    V = self.FFN(tf.concat([X,W],axis=1),str(t+1)+'v')/self.d+self.allones
                    self.Z_history = tf.concat([self.Z_history,Z[:,:,tf.newaxis]],axis=2)
                    self.V_history = tf.concat([self.V_history,V[:,:,tf.newaxis]],axis=2)
            term_delta_1 = Y
            term_delta_2 = tf.abs(S*self.a-self.b*self.T-self.a*W)
            if not self.ispower:
                self.clipped_delta_1 = \
			tf.clip_by_value(term_delta_1, -500.0, 500.0)
                self.clipped_delta_2 = \
			tf.clip_by_value(term_delta_2, -500.0, 500.0)
            if self.ispower:
                self.clipped_delta_1 = \
			tf.clip_by_value(term_delta_1, -5000.0, 5000.0)
                self.clipped_delta_2 = \
			tf.clip_by_value(term_delta_2, -5000.0, 5000.0)
            self.loss = tf.reduce_mean(self.clipped_delta_1**2 + self.clipped_delta_2**2)
        self.t_bd = time.time()-start_time 
    

    def sample_path(self, n_sample):
        dW_sample = np.zeros([n_sample, self.d, self.n_time])
        for i in range(self.n_time):
            dW_sample[:, :, i] = np.reshape(normal.rvs(mean=np.zeros(self.d),cov=1, size=n_sample)*self.sqrt_dtSim,(n_sample, self.d))
        return dW_sample

    def FFN(self, x, name):
        with tf.variable_scope(name):
            
            x_norm = tf.concat([self._batch_norm(x[:,0:-1], name='layer0_normal'),x[:,-1,tf.newaxis]],axis=1)

            layer1 = self.FFN_layer(x_norm, self.n_neuron[1],
                                     name='layer1')
            layer2 = self.FFN_layer(layer1, self.n_neuron[2],
                                     name='layer2')
            z = self.FFN_layer(layer2, self.n_neuron[3],
                                activation_fn=None, name='final')
        return z

    def FFN_layer(self, input_, out_sz, 
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
            # These ops will only be preformed when training
            mean, variance = tf.nn.moments(x, [0], name='moments')
            y = tf.nn.batch_normalization(x, mean, variance, 
                                          beta, gamma, 1e-6)
            y.set_shape(x.get_shape())
            return y


##############################################################################
######################################### 3. Further Functions ###############
##############################################################################

           
def solve_ODEs(s_outstanding,gamma,beta,a,lam,T,atol=1e-6,t_eval=None,dense_output=False,modified=False):
    # y[0]=A[T-t],y[1]=B[T-t],y[2]=C[T-t],y[3]=D[T-t],y[4]=E[T-t],y[5]=F[T-t].
    # Note that the ODE for D is D'(t)=-gamma[1]/(2*lam)*(a+B(s))**2-D(t)*F(t)
    c1 = (gamma[0]-gamma[1])/2.
    c2 = (gamma[0]+gamma[1])/2.
    c3 = c1/(c2*2.)*gamma[1]
    gbar = gamma[0]*gamma[1]/(gamma[0]+gamma[1])
    if modified:
        def rhs_ode(t, y): return [-gamma[1]*s_outstanding/(2.)*(a+y[1])**2+y[2]*y[3]+gbar*(a**2)*s_outstanding,-c1*(beta)*(a+y[1])+y[2]*y[4],-c1*(a+y[1])**2+y[2]*y[5],
                  (gamma[1]*s_outstanding/(2.*lam)*(a+y[1])**2)+y[3]*y[5],-c2/lam*(a+y[1])*beta+y[4]*y[5],-c2/lam*((a+y[1])**2)+(y[5]**2)]           
    else:
        def rhs_ode(t, y): return [(c3*(a+y[1])**2+y[2]*y[3]-gbar*(y[1]**2+2*a*y[1])),-c1*(beta)/a*(a+y[1])**2+y[2]*y[4],-c1*(a+y[1])**2+y[2]*y[5],
                  (gamma[1]/(2.*lam)*(a+y[1])**2)+y[3]*y[5],-c2/lam*(a+y[1])*beta+y[4]*y[5],-c2/lam*((a+y[1])**2)+(y[5]**2)]           
    return solve_ivp(rhs_ode,[0.,T],[0.,0.,0.,0.,0.,0.],t_eval=t_eval,atol=atol,dense_output=dense_output,method='RK23')


def solve_and_compare_ODE(b,s_outstanding,gamma,beta,a,lam,T,dW,x,t_eval):
    solflipped= solve_ODEs(s_outstanding,gamma,beta,a,lam,T,atol=1e-8,t_eval=None,dense_output=True,modified=True)
    n_samples = dW.shape[0]
    phi_array = np.zeros([n_samples,t_eval.shape[0]],dtype=float)
    S_array = np.zeros([n_samples,t_eval.shape[0]],dtype=float)
    W = np.concatenate([np.zeros([n_samples,1,1],dtype=float),np.cumsum(dW,axis=2)],axis=2)
    W_array = np.zeros([n_samples,t_eval.shape[0]],dtype=float)
    dtSim = (T+0.0)/dW.shape[2]
    tSim = np.arange(0, dW.shape[2]+1)*dtSim
    gbar = gamma[0]*gamma[1]/(gamma[0]+gamma[1])
    def sol(t): return solflipped.sol(T-t)
    for i in range(n_samples):
        Wint = scipy.interpolate.interp1d(tSim,W[i,0,:])
        def rhs_ode(t, y): return sol(t)[3]+sol(t)[4]*Wint(t)+sol(t)[5]*y
        temp = solve_ivp(rhs_ode,[0.,T],[x],t_eval=t_eval)
        phi_array[i,:] = temp.y[0,:]
        ABC = sol(t_eval)
        S_frictionless = a*Wint(t_eval)+gbar*(t_eval-T)*(a**2)*s_outstanding+b*T
        S_array[i,:] = S_frictionless+ABC[0,:]+ABC[1,:]*Wint(t_eval)+ABC[2,:]*phi_array[i,:]
        W_array[i,:] = Wint(t_eval)
    return S_array, phi_array, s_outstanding*gamma[1]/(gamma[0]+gamma[1])-W_array                    

def run_Equilibrium(do_training,is_power,dW_test=None):
    tf.reset_default_graph()
    sess=tf.Session()
    tf.set_random_seed(1)
    print("Begin to find Equilibrium")
    if do_training:
        Equilibrium = FBSDESolver(sess,ispower=is_power)
        Equilibrium.build()
        Equilibrium.train() 
        if Equilibrium.model.save:
            saver = tf.train.Saver(tf.trainable_variables())
            save_path = saver.save(sess,Equilibrium.save_dir)
            print("Model saved in file: %s" % save_path)  
    else:
        ex_dict={'load_parameters':True,'n_maxstep':1,'learning_rate':1e-8}
        Equilibrium = FBSDESolver(sess,ex_dict,is_power)       
        Equilibrium.build()
        Equilibrium.train()        
    if dW_test is None:
        dW_test = Equilibrium.sample_path(10000)
    
    feed_dict_test = {Equilibrium.dW: dW_test, Equilibrium.is_training: False, Equilibrium.is_evaluating: False, Equilibrium.Xfull: dW_test}
     
    print(Equilibrium.sess.run(Equilibrium.loss,feed_dict=feed_dict_test))
    
    Z_sample = Equilibrium.sess.run(Equilibrium.Z_history,feed_dict=feed_dict_test)
    X_sample = Equilibrium.sess.run(Equilibrium.X_history,feed_dict=feed_dict_test)
    Y_sample = Equilibrium.sess.run(Equilibrium.Y_history,feed_dict=feed_dict_test)
    S_sample = Equilibrium.sess.run(Equilibrium.S_history,feed_dict=feed_dict_test)
    V_sample = Equilibrium.sess.run(Equilibrium.V_history,feed_dict=feed_dict_test)

    s_outstanding, gamma,beta,a,b,lam,T,x = Equilibrium.s_outstanding, Equilibrium.gamma,Equilibrium.model.beta[0],Equilibrium.a,Equilibrium.b,Equilibrium.model.lam,Equilibrium.T,Equilibrium.phiini
    t_eval = Equilibrium.tSim
    stradd = Equilibrium.model.stradd  

                    
    return dW_test,Z_sample,X_sample,Y_sample,S_sample,V_sample,s_outstanding, gamma,beta,a,b,lam,T,x,t_eval,stradd
    


##############################################################################
######################################### 4. Run FBSDE Solver ################
##############################################################################



if __name__ == '__main__':
    do_training = False #set to False to load previously trained networks
    create_csv = False # set to True to create csv files with the data
    np.random.seed(1)
    #### Solve the FBSDEs for both quadratic and 3/2 costs####       
    dW_test,Z_sample,X_sample,Y_sample,S_sample,V_sample,s_outstanding, gamma,beta,a,b,lam,T,x,t_eval,stradd = run_Equilibrium(do_training,False)   
    dW_test,Z_sample2,X_sample2,Y_sample2,S_sample2,V_sample2,s_outstanding2, gamma2,beta2,a2,b2,lam2,T2,x2,t_eval2,stradd2 = run_Equilibrium(do_training,True,dW_test=dW_test)   
 
        
    #### Solve the Riccati ODEs (benchmark in the quadratic case)
    res = solve_ODEs(s_outstanding,gamma,beta,a,lam,T,atol=1e-6,t_eval=t_eval,dense_output=False,modified=True)
    S,phi,phinofric = solve_and_compare_ODE(b,s_outstanding,gamma,beta,a,lam,T,dW_test,x,t_eval)   
       # W = -(phinofric-s_outstanding*gamma[1]/(gamma[0]+gamma[1]))
    
    #### Transform increments of Brownian motion to sample paths on appropriate grid
    W = np.concatenate([np.zeros([dW_test.shape[0],1,1],dtype=float),np.cumsum(dW_test,axis=2)],axis=2)
    W_array = np.zeros([dW_test.shape[0],t_eval.shape[0]],dtype=float)
    dtSim = (T+0.0)/dW_test.shape[2]
    tSim = np.arange(0, dW_test.shape[2]+1)*dtSim
    for i in range(dW_test.shape[0]):
        Wint = scipy.interpolate.interp1d(tSim,W[i,0,:])
        W_array[i,:] = Wint(t_eval)
    W = W_array
    
    #### Frictionless equilibrium price
    Sfrictionless = b*T+a*W+s_outstanding*gamma[0]*gamma[1]/(gamma[0]+gamma[1])*(t_eval-T)*(a**2)

    #### Generate Plots
    for ind in [5]:
        plt.figure()
        plt.plot(t_eval,S_sample2[ind,0,:]*a-Sfrictionless[ind,:],label='samplePath1')
        plt.plot(t_eval,S_sample2[ind+2,0,:]*a-Sfrictionless[ind+2,:],label='samplePath2')
        plt.legend()
        axes = plt.gca()
        axes.set_ylim([-0.2,0.2])
        plt.xlabel(r'$t$')
        plt.ylabel('Difference at $t$')  
        plt.savefig('Sdiff' + str(ind)+ stradd2 + '.pdf')
        plt.figure()
        plt.plot(t_eval,S_sample[ind,0,:]*a-Sfrictionless[ind,:],label='samplePath1')
        plt.plot(t_eval,S_sample[ind+2,0,:]*a-Sfrictionless[ind+2,:],label='samplePath2')
        plt.plot(t_eval,S[ind,:]-Sfrictionless[ind,:],label='odePath1')
        plt.plot(t_eval,S[ind+2,:]-Sfrictionless[ind+2,:],label='odePath2')
        plt.legend()
        axes = plt.gca()
        axes.set_ylim([-0.2,0.2])
        plt.xlabel(r'$t$')
        plt.ylabel('Difference at $t$')  
        plt.savefig('Sdiff' + str(ind)+ stradd + '.pdf')
        if create_csv:
            np.savetxt('Figure5tgrid.csv',t_eval,delimiter=',')
            np.savetxt('Figure5RightPanelsamplePath1.csv',S_sample2[ind,0,:]*a-Sfrictionless[ind,:],delimiter=',')
            np.savetxt('Figure5RightPanelsamplePath2.csv',S_sample2[ind+2,0,:]*a-Sfrictionless[ind+2,:],delimiter=',')
            np.savetxt('Figure5LeftPanelsamplePath1.csv',S_sample[ind,0,:]*a-Sfrictionless[ind,:],delimiter=',')
            np.savetxt('Figure5LeftPanelsamplePath2.csv',S_sample[ind+2,0,:]*a-Sfrictionless[ind+2,:],delimiter=',')
            np.savetxt('Figure5LeftPanelODEPath1.csv',S[ind,:]-Sfrictionless[ind,:],delimiter=',')
            np.savetxt('Figure5LeftPanelODEPath2.csv',S[ind+2,:]-Sfrictionless[ind+2,:],delimiter=',') 


    
    plt.figure()
    plt.plot(t_eval[:-1],V_sample[ind,0,:]*a-a,label='samplePath1')
    plt.plot(t_eval[:-1],V_sample[ind+1,0,:]*a-a,label='samplePath2')
    plt.plot(t_eval,np.flip(res.y[1,:]),label='ode')
    plt.legend()
    axes = plt.gca()
    axes.set_ylim([0.,0.05])
    plt.xlabel(r'$t$')
    plt.ylabel('Difference at $t$')  
    plt.savefig('sigmadiff' + stradd + '.pdf')    
    plt.figure()
    plt.plot(t_eval[:-1],V_sample2[ind,0,:]*a-a,label='samplePath1')
    plt.plot(t_eval[:-1],V_sample2[ind+1,0,:]*a-a,label='samplePath2')
    plt.legend()
    axes = plt.gca()
    axes.set_ylim([0.,0.05])
    plt.xlabel(r'$t$')
    plt.ylabel('Difference at $t$')  
    plt.savefig('sigmadiff' + stradd2 + '.pdf') 
    if create_csv:
        np.savetxt('Figure6tgrid.csv',t_eval[:-1],delimiter=',')
        np.savetxt('Figure6RightPanelsamplePath1.csv',V_sample2[ind,0,:]*a-a,delimiter=',')
        np.savetxt('Figure6RightPanelsamplePath2.csv',V_sample2[ind+1,0,:]*a-a,delimiter=',')
        np.savetxt('Figure6LeftPanelsamplePath1.csv',V_sample[ind,0,:]*a-a,delimiter=',')
        np.savetxt('Figure6LeftPanelsamplePath2.csv',V_sample[ind+1,0,:]*a-a,delimiter=',')
        np.savetxt('Figure6tgridODE.csv',t_eval,delimiter=',')
        np.savetxt('Figure6LeftPanelODE.csv',np.flip(res.y[1,:]),delimiter=',')

 
    
    ## histograms at t = 10 and t = 19.5
    for i in [20,39]:
        plt.figure()
        bins = np.linspace(-0.005,0.005,40)
        data_plot = [V_sample[:,0,i]*a-a-np.flip(res.y[1,:])[i+1],S[:,i]-S_sample[:,0,i]*a]
        plt.hist(data_plot,bins,alpha=0.5,label=['sigma','S'])
        plt.legend(loc='upper left')
        plt.savefig('approximationErrorsQuadratic'+'Hist'+str(i)+'.pdf',bbox_inches ='tight')
        if create_csv:
            np.savetxt('Figure7Vol'+str(i)+'.csv',V_sample[:,0,i]*a-a-np.flip(res.y[1,:])[i+1],delimiter=',')
            np.savetxt('Figure7Price'+str(i)+'.csv',S[:,i]-S_sample[:,0,i]*a,delimiter=',')


        plt.figure()
        if i == 20:
            bins= np.linspace(-0.5,0.5,40)
        else:
            bins = np.linspace(-0.05,0.05,40)
        data_plot = [S_sample[:,0,i]*a-Sfrictionless[:,i],S_sample2[:,0,i]*a-Sfrictionless[:,i]]
        plt.hist(data_plot,bins,alpha=0.5,label=['Quadratic cost','3/2 cost'])
        plt.legend(loc='upper left')
        plt.savefig('Sdiff'+'hist'+str(i)+'.pdf',bbox_inches ='tight')
        if create_csv:
            np.savetxt('Figure8Quadratic'+str(i)+'.csv',S_sample[:,0,i]*a-Sfrictionless[:,i],delimiter=',')
            np.savetxt('Figure8Pow'+str(i)+'.csv',S_sample2[:,0,i]*a-Sfrictionless[:,i],delimiter=',')
            
        plt.figure()
        bins = np.linspace(-0.05,0.05,40)
        plt.hist(S_sample[:,0,i]*a-S_sample2[:,0,i]*a,bins,alpha=0.5)
        plt.savefig('SAdjDiff'+'hist'+str(i)+'.pdf',bbox_inches ='tight')
        if create_csv:
            np.savetxt('SAdjDiff'+str(i)+'.csv',S_sample[:,0,i]*a-S_sample2[:,0,i]*a,delimiter=',')
       
