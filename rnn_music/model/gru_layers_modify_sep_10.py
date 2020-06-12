
import numpy as np
import theano
import theano.tensor as tensor
from utils import _p
from utils import ortho_weight, uniform_weight, zero_bias

from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
    
""" Decoder using GRU Recurrent Neural Network. """

trng = RandomStreams(123)

def v_dk(beta):
    betas = np.array([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    vs = np.array([-.33, -.472, -.631, -.792, -.953, -1.11, -1.29, -1.49, -1.74, -2.10, -10])
    from scipy.interpolate import interp1d
    return interp1d(betas, vs)(beta)
'''
def sample(mu, sig):
    
    r = trng.normal(mu.eval().shape, avg = 0.0, std = 1.0, 
                      dtype=theano.config.floatX)
    #sig = 10 ** nu
    return r * sig + mu
'''
def sample(mu, nu):
    
    r = trng.normal(mu.eval().shape, avg = 0.0, std = 1.0, 
                      dtype=theano.config.floatX)
    sig = 10 ** nu
    return r * sig + mu
def param_init_decoder_modify_sep_10(beta, options, params, prefix='decoder_gru'):
    
    n_x = options['n_x']
    n_h = options['n_h']
    
    ################## W ##################################
    W = np.concatenate([uniform_weight(n_x,n_h),
                        uniform_weight(n_x,n_h)], axis=1)    
    W_combined = np.zeros([2]+list(W.shape))
    W_combined[0, :] = W
    #W_combined[1, :] = np.random.randn(np.product(W.shape)).reshape(W.shape) * .001
    W_combined[1, :] = v_dk(beta)
    params[_p(prefix,'W')] = W_combined
    ################## W ##################################
    
    ################## U ######################
    U = np.concatenate([ortho_weight(n_h),
                        ortho_weight(n_h)], axis=1)
    U_combined = np.zeros([2]+list(U.shape))
    U_combined[0, :] = U
    #U_combined[1, :] = np.random.randn(np.product(U.shape)).reshape(U.shape) * .001
    U_combined[1, :] = v_dk(beta)
    params[_p(prefix,'U')] = U_combined
    ################## U ######################
    
    ################## b ##################################
    b = zero_bias(2*n_h)
    b_combined = np.zeros([2]+list(b.shape))
    b_combined[0, :] = b
    #b_combined[1, :] = np.random.randn(np.product(b.shape)).reshape(b.shape) * .001
    b_combined[1, :] = v_dk(beta)
    params[_p(prefix,'b')] = b_combined
    ################## b ##################################
    
    ################## Wx #####################
    Wx = uniform_weight(n_x, n_h)
    Wx_combined = np.zeros([2]+list(Wx.shape))
    Wx_combined[0, :] = Wx
    #Wx_combined[1, :] = np.random.randn(np.product(Wx.shape)).reshape(Wx.shape) * .001
    Wx_combined[1, :] = v_dk(beta)
    params[_p(prefix,'Wx')] = Wx_combined
    ################## Wx #####################
    
    ################## Ux #################################
    Ux = ortho_weight(n_h)
    Ux_combined = np.zeros([2]+list(Ux.shape))
    Ux_combined[0, :] = Ux
    #Ux_combined[1, :] = np.random.randn(np.product(Ux.shape)).reshape(Ux.shape) * .001
    Ux_combined[1, :] = v_dk(beta)
    params[_p(prefix,'Ux')] = Ux_combined
    ################## Ux #################################
    
    ################## bx #####################
    bx = zero_bias(n_h)
    bx_combined = np.zeros([2]+list(bx.shape))
    bx_combined[0, :] = bx
    #bx_combined[1, :] = np.random.randn(np.product(bx.shape)).reshape(bx.shape) * .001
    bx_combined[1, :] = v_dk(beta)
    params[_p(prefix,'bx')] = bx_combined
    ################## bx #####################
    
    ################## b0 #################################
    b0 = zero_bias(n_h)
    b0_combined = np.zeros([2]+list(b0.shape))
    b0_combined[0, :] = b0
    #b0_combined[1, :] = np.random.randn(np.product(b0.shape)).reshape(b0.shape) * .001
    b0_combined[1, :] = v_dk(beta)
    params[_p(prefix,'b0')] = b0_combined
    ################## b0 #################################

    return params   
    

def decoder_layer_modify_sep_10(tparams, state_below, prefix='decoder_gru'):
    
    """ state_below: size of n_steps *  n_x 
    """

    ###############################################################################
    
    #tparams = theano.printing.Print('tparams: ')(tparams)
    
    tparams_W = sample(tparams[_p(prefix, 'W')][0], tparams[_p(prefix, 'W')][1])
    tparams_U = sample(tparams[_p(prefix, 'U')][0], tparams[_p(prefix, 'U')][1])
    tparams_b = sample(tparams[_p(prefix, 'b')][0], tparams[_p(prefix, 'b')][1])
    tparams_Wx = sample(tparams[_p(prefix, 'Wx')][0], tparams[_p(prefix, 'Wx')][1])
    tparams_Ux = sample(tparams[_p(prefix, 'Ux')][0], tparams[_p(prefix, 'Ux')][1])
    tparams_bx = sample(tparams[_p(prefix, 'bx')][0], tparams[_p(prefix, 'bx')][1])
    tparams_b0 = sample(tparams[_p(prefix, 'b0')][0], tparams[_p(prefix, 'b0')][1])
    
    #tparams_b0 = theano.printing.Print('tparams_b0: ')(tparams_b0)
    ###############################################################################

    n_steps = state_below.shape[0]
    n_h = tparams[_p(prefix,'Ux')].shape[1]
        
    state_belowx0 = tparams_b0
    h0vec = tensor.tanh(state_belowx0)
    h0 = h0vec.dimshuffle('x',0)
    
    def _slice(_x, n, dim):
        return _x[n*dim:(n+1)*dim]
        
    state_below_ = tensor.dot(state_below, tparams_W)  + tparams_b
    state_belowx = tensor.dot(state_below, tparams_Wx) + tparams_bx
    
    def _step_slice(x_, xx_, h_, U, Ux):
        preact = tensor.dot(h_, U)
        preact += x_

        r = tensor.nnet.sigmoid(_slice(preact, 0, n_h))
        u = tensor.nnet.sigmoid(_slice(preact, 1, n_h))

        preactx = tensor.dot(h_, Ux)
        preactx = preactx * r
        preactx = preactx + xx_

        h = tensor.tanh(preactx)

        h = u * h_ + (1. - u) * h

        return h
    
    seqs = [state_below_[:n_steps-1], state_belowx[:n_steps-1]]
    _step = _step_slice

    rval, updates = theano.scan(_step,
                                sequences=seqs,
                                outputs_info = [h0vec],
                                non_sequences = [tparams_U,
                                                 tparams_Ux],
                                name=_p(prefix, '_layers'),
                                n_steps=n_steps-1)
                                
    #h0x = h0.dimshuffle('x',0,1)
                            
    return tensor.concatenate((h0,rval))
