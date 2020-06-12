import theano
import theano.tensor as tensor
import numpy as np
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
#from theano.tensor.shared_randomstreams import RandomStreams
from utils import numpy_floatX

from utils import _p
 
import pdb
from theano.compile.nanguardmode import NanGuardMode

def SGD(tparams, cost, inps, lr, clip_norm=5):
    
    grads = tensor.grad(cost, tparams.values())
    norm = tensor.sqrt(sum([tensor.sum(g**2) for g in grads]))
    if tensor.ge(norm, clip_norm):
        grads = [g*clip_norm/norm for g in grads]
        
    gshared = [theano.shared(p.get_value() * 0., name='%s_grad'%k) 
                for k, p in tparams.iteritems()]
    gsup = [(gs, g) for gs, g in zip(gshared, grads)]
    f_grad_shared = theano.function(inps, cost, updates=gsup)
    
    updates = []

    for p, g in zip(tparams.values(), gshared):        
        updated_p = p - lr * g
        updates.append((p, updated_p))
    
    f_update = theano.function([lr], [], updates=updates)
    
    return f_grad_shared, f_update 
    
def Momentum(tparams, cost, inps, lr, momentum=0.9, clip_norm=5):
    
    grads = tensor.grad(cost, tparams.values())
    norm = tensor.sqrt(sum([tensor.sum(g**2) for g in grads]))
    if tensor.ge(norm, clip_norm):
        grads = [g*clip_norm/norm for g in grads]
        
    gshared = [theano.shared(p.get_value() * 0., name='%s_grad'%k) 
                for k, p in tparams.iteritems()]
    gsup = [(gs, g) for gs, g in zip(gshared, grads)]
    f_grad_shared = theano.function(inps, cost, updates=gsup) 
    
    updates = []

    for p, g in zip(tparams.values(), gshared): 
        m = theano.shared(p.get_value() * 0.)
        m_new = momentum * m - lr * g
        updates.append((m, m_new))        
        
        updated_p = p + m_new
        updates.append((p, updated_p))
    
    f_update = theano.function([lr], [], updates=updates)
    
    return f_grad_shared, f_update 
    
def RMSprop(tparams, cost, inps, lr, rho=0.9, epsilon=1e-6, clip_norm=5):
    
    grads = tensor.grad(cost, tparams.values())
    norm = tensor.sqrt(sum([tensor.sum(g**2) for g in grads]))
    if tensor.ge(norm, clip_norm):
        grads = [g*clip_norm/norm for g in grads]
        
    gshared = [theano.shared(p.get_value() * 0., name='%s_grad'%k) 
                for k, p in tparams.iteritems()]
    gsup = [(gs, g) for gs, g in zip(gshared, grads)]
    f_grad_shared = theano.function(inps, cost, updates=gsup)     
    
    updates = []

    for p, g in zip(tparams.values(), gshared):
        acc = theano.shared(p.get_value() * 0.)
        acc_new = rho * acc + (1 - rho) * g ** 2
        updates.append((acc, acc_new))
        
        updated_p = p - lr * (g / tensor.sqrt(acc_new + epsilon))
        updates.append((p, updated_p))
    
    f_update = theano.function([lr], [], updates=updates)
    
    return f_grad_shared, f_update
      
def Adam(tparams, cost, inps, lr, b1=0.1, b2=0.001, e=1e-8, clip_norm=5):
    
    grads = tensor.grad(cost, tparams.values())
    norm = tensor.sqrt(sum([tensor.sum(g**2) for g in grads]))
    if tensor.ge(norm, clip_norm):
        grads = [g*clip_norm/norm for g in grads]
    
    gshared = [theano.shared(p.get_value() * 0., name='%s_grad'%k) 
                for k, p in tparams.iteritems()]
    gsup = [(gs, g) for gs, g in zip(gshared, grads)]
    f_grad_shared = theano.function(inps, cost, updates=gsup)
    
    updates = []

    i = theano.shared(numpy_floatX(0.))    
    i_t = i + 1.
    fix1 = 1. - b1**(i_t)
    fix2 = 1. - b2**(i_t)
    lr_t = lr * (tensor.sqrt(fix2) / fix1)

    for p, g in zip(tparams.values(), gshared):
        m = theano.shared(p.get_value() * 0.)
        v = theano.shared(p.get_value() * 0.)
        m_t = (b1 * g) + ((1. - b1) * m)
        v_t = (b2 * tensor.sqr(g)) + ((1. - b2) * v)
        g_t = m_t / (tensor.sqrt(v_t) + e)
        p_t = p - (lr_t * g_t)
        updates.append((m, m_t))
        updates.append((v, v_t))
        updates.append((p, p_t))
    updates.append((i, i_t))
    
    f_update = theano.function([lr], [], updates=updates)
    
    return f_grad_shared, f_update  
    
def Santa_r(tparams, cost, inps, lr, rho=0.95, e=1e-8, clip_norm=5):
    """ The implementation of Santa algorithm running on the refinement stage,
        but also update \alpha as done in the exploration stage.
        tparams: theano shared variables, params that we need to optimize
        cost: cost function, the cross-entropy loss in our case
        inps: input theano variables
        lr: learning rate, in our case, we choose it to be 1.*1e-3, or 2.*1e-4
        rho, e, clip_norm: hyper-parameters we used in all the algorithms.
    """
    
    grads = tensor.grad(cost, tparams.values())
    norm = tensor.sqrt(sum([tensor.sum(g**2) for g in grads]))
    if tensor.ge(norm, clip_norm):
        grads = [g*clip_norm/norm for g in grads]
    
    gshared = [theano.shared(p.get_value() * 0., name='%s_grad'%k) 
                for k, p in tparams.iteritems()]
    gsup = [(gs, g) for gs, g in zip(gshared, grads)]
    f_grad_shared = theano.function(inps, cost, updates=gsup)
    
    updates = []

    for p, g in zip(tparams.values(), gshared):
        m = theano.shared(p.get_value() * 0.)
        v = theano.shared(p.get_value() * 0.)
        alpha = theano.shared(np.ones(p.get_value().shape)*.5)
        
        alpha_t = alpha + m**2
        v_t = rho * v + (1.-rho) * (g ** 2) 
        pcder = tensor.sqrt(tensor.sqrt(v_t)+e)    
        
        m_t = -lr*g/pcder + ((1. - alpha_t) * m)
        p_t = p + (m_t/ pcder)
        
        updates.append((alpha, alpha_t))
        updates.append((m, m_t))
        updates.append((v, v_t))
        updates.append((p, p_t))
    
    f_update = theano.function([lr], [], updates=updates)
    
    return f_grad_shared, f_update 
    
def Santa_modify(beta, batchsize, tparams, cost, inps, lr, eidx, nframes, max_epoch, rho=0.95, anne_rate=0.5, e=1e-8, clip_norm=5):
    """ The implementation of Santa algorithm.
        tparams: theano shared variables, params that we need to optimize
        cost: cost function, the cross-entropy loss in our case
        inps: input theano variables
        lr: learning rate, in our case, we choose it to be 1.*1e-3, or 2.*1e-4
        eidx: the current epochs we are running, used to decide when to change 
            from exploration to refinement
        nframes: how many time-steps we have in the training dataset.
        max_epoch: the maximum of epochs we run
        rho, anne_rate, e, clip_norm: hyper-parameters we used in all the algorithms.
    """
    
    #theano.config.compute_test_value = 'warn'
    
    trng = RandomStreams(123)
    #pdb.set_trace()
    grads = tensor.grad(cost, tparams.values())
    #grads_b0 = theano.printing.Print('grads_b0: ')(tensor.grad(cost, tparams['decoder_gru_b0']))
    
    #grads_b0 = tensor.grad(cost, tparams['decoder_gru_b0'])
    #get_grad_b0 = theano.function([inps, cost], grads_b0)
    #pdb.set_trace()
    
    norm = tensor.sqrt(sum([tensor.sum(g**2) for g in grads]))
    if tensor.ge(norm, clip_norm):
        grads = [g*clip_norm/norm for g in grads]
    
    gshared = [theano.shared(p.get_value() * 0., name='%s_grad'%k) 
                for k, p in tparams.iteritems()]
    gsup = [(gs, g) for gs, g in zip(gshared, grads)]
    #f_grad_shared = theano.function(inps, cost, updates=gsup, 
    #                                mode=NanGuardMode(nan_is_error=True, inf_is_error=True, big_is_error=True))
    
    f_grad_shared = theano.function(inps, cost, updates=gsup)
    
    updates = []
    
    i = theano.shared(numpy_floatX(0.))    
    i_t = i + 1.

    pcder_list = []
    #for p, g in zip(tparams.values(), gshared):
    for w, g in zip(tparams.values(), gshared):
        #m = theano.shared(p.get_value() * 0.)
        m = theano.shared(w.get_value() * 0.)
        #v = theano.shared(p.get_value() * 0.)
        v = theano.shared(w.get_value() * 0.)
        alpha = theano.shared(np.ones(w.get_value().shape)*.5)
        
        ######################## initialize w #################################
        #w_numpy = np.zeros([2] + p.shape.eval().tolist())
        #w_numpy[0, :] = p.eval()
        #w_numpy[1, :] = np.random.normal(size=p.shape.eval()) ** 0.001
        #w = theano.shared(w_numpy)
        ######################## initialize w #################################
        
        alpha_t = alpha + (m**2 - lr/(i_t ** anne_rate)) * tensor.lt(eidx, 0.15*max_epoch) 
        v_t = rho * v + (1.-rho) * (g ** 2) 
        pcder = tensor.sqrt(tensor.sqrt(v_t)+e) 
            
        #eps = trng.normal(p.get_value().shape, avg = 0.0, std = 1.0, 
        #                  dtype=theano.config.floatX)
        eps = trng.normal(w.get_value().shape, avg = 0.0, std = 1.0, 
                          dtype=theano.config.floatX)
        
        ######################## main change ########################################################
        #################### Lhat(w, wbar, r, beta) ####################################
        # logq(wbar,z=sample(w,r)):
        def get_mu_sig(wbar1):
            mu11 = wbar1[0, :]
            sig11 = 10 ** wbar1[1, :]
            return mu11, sig11
        def sample(w2, r2):
            mu2, sig2 = get_mu_sig(w2)
            return r2 * sig2 + mu2
        def logq(wbar1, z1):
            mu1, sig1 = get_mu_sig(wbar1)
            logsig1 = wbar1[1, :] * tensor.log(10)
            return tensor.mean(tensor.sum(-((z1 - mu1)/sig1) ** 2 / 2 - .5 * tensor.log(2*np.pi) - logsig1, axis=0))
        def ddw_Lhat_wo_Ez_wbar(wbar):                
            #NOTE: unlike Domke - we put batchsize first in shape of r
            r = trng.normal([batchsize]+list(w.get_value().shape), avg = 0.0, std = 1.0, dtype=theano.config.floatX)
            z = sample(w, r)
            return tensor.grad( logq(wbar, z), w )
        #################### Lhat(w, wbar, r, beta) ####################################
        
        ################# logprior_o_w_wq #####################################
        # logprior(w, w_q):
        def v_dk(beta):
            betas = np.array([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
            vs = np.array([-.33, -.472, -.631, -.792, -.953, -1.11, -1.29, -1.49, -1.74, -2.10, -10])
            from scipy.interpolate import interp1d
            return interp1d(betas, vs)(beta)
        q_sig2 = 1.000
        logprior_o_w_wq = tensor.sum( -(w[1, :] - v_dk(beta)) ** 2 / (2*q_sig2) - .5 * tensor.log(2*np.pi * q_sig2) )
        ################# logprior_o_w_wq #####################################
        
        ######################################################
        ddw_Lhat_wo_Ez = ddw_Lhat_wo_Ez_wbar(w)
        ddw_logprior_o_w_wq = tensor.grad(logprior_o_w_wq, w)
        ######################################################
        
        g1 = g + (beta - 1) * ddw_Lhat_wo_Ez + beta * ddw_logprior_o_w_wq 
        ######################## main change ########################################################
        
        m_t = -lr*g1/pcder + (1. - alpha_t) * m + (tensor.sqrt(2*lr*v_t/(i_t ** anne_rate)/nframes) *eps) * tensor.lt(eidx, 0.15*max_epoch)
        w_t = w + (m_t/ pcder)
        #pcder_print = theano.printing.Print('pcder is: ')(pcder)
        #f2 = theano.function([pcder], [pcder_print])
        #w_t = w + m_t
        updates.append((alpha, alpha_t))
        updates.append((m, m_t))
        updates.append((v, v_t))
        updates.append((w, w_t))
        pcder_list.append(pcder)
    updates.append((i, i_t))
    #f_update = theano.function([lr,eidx,nframes,max_epoch], [], updates=updates, 
    #                           mode=NanGuardMode(nan_is_error=True, inf_is_error=True, big_is_error=True))
    
    f_update = theano.function([lr,eidx,nframes,max_epoch], [w, ddw_Lhat_wo_Ez, ddw_logprior_o_w_wq,  g, alpha, m, v, w, pcder], updates=updates)    
    return f_grad_shared, f_update
    
