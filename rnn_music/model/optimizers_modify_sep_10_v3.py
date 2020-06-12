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
    
    trng = RandomStreams(123)

    for p, g in zip(tparams.values(), gshared):
        #noise_langevin = tensor.sqrt(lr) * trng.normal(p.get_value().shape, avg = 0.0, std = 1.0, dtype=theano.config.floatX)
        #pdb.set_trace()
        noise_langevin = trng.normal(np.zeros(p.get_value().shape).shape, avg = 0.0, std = 0.001, dtype=theano.config.floatX)
        updated_p = p - lr * g + noise_langevin
        updates.append((p, updated_p))
    #pdb.set_trace()
    f_update = theano.function([], [], updates=updates)
    
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
    
def Santa_modify(beta, v_dk_beta, batchsize, tparams, cost, inps, lr, eidx, nframes, max_epoch, rho=0.95, anne_rate=0.5, e1=1e-8, e=1e-8, clip_norm=2):
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
        w_dummy = np.zeros(w.get_value().shape)[0, :]
        #m = theano.shared(p.get_value() * 0.)
        #m = theano.shared(w.get_value() * 0.)
        #v = theano.shared(p.get_value() * 0.)
        v = theano.shared(w.get_value() * 0.)
        alpha = theano.shared(np.ones(w.get_value().shape)*.5)
        
        ######################## main change ########################################################
        
        nu = w[1, :]
        
        r = trng.normal([batchsize] + list(w_dummy.shape), avg = 0.0, std = 1.0, dtype=theano.config.floatX)        
        #pdb.set_trace()
        
        dim = (1,) + w_dummy.shape
        ddw_Lhat_wo_Ez_mu = tensor.reshape(tensor.mean(r, axis=0), dim) # d_dmu
        ddw_Lhat_wo_Ez_nu = tensor.reshape(tensor.mean(r**2 * (10**nu) * np.log(10), axis=0), dim) # d_dnu
        ddw_Lhat_wo_Ez = tensor.concatenate((ddw_Lhat_wo_Ez_mu, ddw_Lhat_wo_Ez_nu), axis=0)
        
        
        ddw_logprior_o_w_wq_mu = tensor.reshape(w_dummy * 0, dim)
        ddw_logprior_o_w_wq_nu = tensor.reshape(nu - v_dk_beta, dim)
        ddw_logprior_o_w_wq = tensor.concatenate((ddw_logprior_o_w_wq_mu, ddw_logprior_o_w_wq_nu), axis=0)
        
        cap_g = 1.7
        norm_g = tensor.sqrt(g**2) + e
        if tensor.ge(norm_g, cap_g):
            g = g * cap_g/norm_g
        
        cap_ddw_Lhat_wo_Ez = 1.7
        norm_ddw_Lhat_wo_Ez = tensor.sqrt(ddw_Lhat_wo_Ez**2) + e
        if tensor.ge(norm_ddw_Lhat_wo_Ez, cap_ddw_Lhat_wo_Ez):
            ddw_Lhat_wo_Ez = ddw_Lhat_wo_Ez * cap_ddw_Lhat_wo_Ez/norm_ddw_Lhat_wo_Ez
        
        cap_ddw_logprior_o_w_wq = 1.7
        norm_ddw_logprior_o_w_wq = tensor.sqrt(ddw_logprior_o_w_wq**2) + e
        if tensor.ge(norm_ddw_logprior_o_w_wq, cap_ddw_logprior_o_w_wq):
            ddw_logprior_o_w_wq = ddw_logprior_o_w_wq * cap_ddw_logprior_o_w_wq/norm_ddw_logprior_o_w_wq
        
        cap_g1 = 5
        g1 = g + (beta - 1) * ddw_Lhat_wo_Ez + beta * ddw_logprior_o_w_wq
        norm_g1 = tensor.sqrt(g1**2) + e
        if tensor.ge(norm_g1, cap_g1):
            g1 = g1 * cap_g1/norm_g1
        ######################## main change ########################################################
        
        eps = trng.normal(w.get_value().shape, avg = 0.0, std = 1.0, 
                          dtype=theano.config.floatX)
        lr_anne = lr/(i_t ** anne_rate)
        v_t = rho * v + (1.-rho) * (g ** 2) 
        pcder = tensor.sqrt(tensor.sqrt(v_t)+e1) 
        grad =  lr_anne * g1/pcder + tensor.sqrt(2*lr_anne/pcder) * eps / nframes * tensor.lt(eidx, 0.15*max_epoch)
        
        w_t = w - grad
        
        updates.append((v, v_t))
        updates.append((w, w_t))
        pcder_list.append(pcder)
    updates.append((i, i_t))

    #f_update = theano.function([lr,eidx,nframes,max_epoch], [ddw_Lhat_wo_Ez, ddw_logprior_o_w_wq,  g, alpha, m, v, w, pcder], updates=updates)
    f_update = theano.function([lr,eidx,nframes,max_epoch], [ddw_Lhat_wo_Ez, ddw_logprior_o_w_wq,  g, alpha, v, w, pcder], updates=updates)
    return f_grad_shared, f_update
    
