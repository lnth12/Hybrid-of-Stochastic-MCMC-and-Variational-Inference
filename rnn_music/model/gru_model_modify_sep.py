import numpy as np
import theano
import theano.tensor as tensor
from theano import config

from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

from collections import OrderedDict
#from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

#from utils import dropout, numpy_floatX
from utils import uniform_weight, zero_bias

from gru_layers_modify_sep import param_init_decoder_modify, decoder_layer_modify

import pdb

trng = RandomStreams(123)

# Set the random number generators' seeds for consistency
SEED = 123  
np.random.seed(SEED)

def v_dk(beta):
    betas = np.array([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    vs = np.array([-.33, -.472, -.631, -.792, -.953, -1.11, -1.29, -1.49, -1.74, -2.10, -10])
    from scipy.interpolate import interp1d
    return interp1d(betas, vs)(beta)

""" init. parameters. """  
def init_params_modify(options, beta):
# changed    
    n_x = options['n_x']  
    n_h = options['n_h']
    
    params = OrderedDict()
    params = param_init_decoder_modify(beta, options,params)
    
    Vhid = uniform_weight(n_h,n_x)
    Vhid_combined = np.zeros([2]+list(Vhid.shape))
    Vhid_combined[0, :] = Vhid
    #Vhid_combined[1, :] = np.random.randn(np.product(Vhid.shape)).reshape(Vhid.shape) * .001
    Vhid_combined[1, :] = v_dk(beta)
    params['Vhid'] = Vhid_combined
    
    bhid = zero_bias(n_x)
    bhid_combined = np.zeros([2]+list(bhid.shape))
    bhid_combined[0, :] = bhid
    #bhid_combined[1, :] = np.random.randn(np.product(bhid.shape)).reshape(bhid.shape) * .001
    bhid_combined[1, :] = v_dk(beta)
    params['bhid'] = bhid_combined                                  

    return params

def init_tparams(params):
# no change
    tparams = OrderedDict()
    for kk, pp in params.iteritems():
        tparams[kk] = theano.shared(params[kk], name=kk)
    return tparams
    
""" Building model... """

def build_model_modify(tparams,options):
    
    #trng = RandomStreams(SEED)
    
    # Used for dropout.
    #use_noise = theano.shared(numpy_floatX(0.))

    # x: n_steps * n_x
    x = tensor.matrix('x', dtype=config.floatX)      
    n_steps = x.shape[0]                                                                              
                                             
    h_decoder = decoder_layer_modify(tparams, x)
    
    #h_decoder_printed = theano.printing.Print('h_decoder: ')(h_decoder)
    
    ###############################################################################
    def sample(mu, nu):
        
        r = trng.normal(mu.eval().shape, avg = 0.0, std = 1.0, 
                          dtype=theano.config.floatX)
        sig = 10 ** nu
        return r * sig + mu
    
    tparams_Vhid = sample(tparams['Vhid'][0], tparams['Vhid'][1])
    tparams_bhid = sample(tparams['bhid'][0], tparams['bhid'][1])
    pdb.set_trace()
    ###############################################################################    
    
    #pred = tensor.nnet.sigmoid(tensor.dot(h_decoder,tparams['Vhid']) + tparams['bhid'])
    pred = tensor.nnet.sigmoid(tensor.dot(h_decoder,tparams_Vhid) + tparams_bhid)
    
    #printed_pred = theano.printing.Print('pred: ')(pred)
    
    f_pred = theano.function([x], pred)
    
    cost = tensor.sum(tensor.nnet.binary_crossentropy(pred, x))/n_steps                         

    return x, f_pred, cost
