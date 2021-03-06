import theano
import theano.tensor as tensor
import numpy as np
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
#from theano.tensor.shared_randomstreams import RandomStreams
from utils import numpy_floatX
    
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
    
def Santa(tparams, cost, inps, lr, eidx, nframes, max_epoch, rho=0.95, anne_rate=0.5, e=1e-8, clip_norm=5):
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
    
    trng = RandomStreams(123)
    
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

    for p, g in zip(tparams.values(), gshared):
        m = theano.shared(p.get_value() * 0.)
        v = theano.shared(p.get_value() * 0.)
        alpha = theano.shared(np.ones(p.get_value().shape)*.5)
        
        alpha_t = alpha + (m**2 - lr/(i_t ** anne_rate)) * tensor.lt(eidx, 0.15*max_epoch) 
        v_t = rho * v + (1.-rho) * (g ** 2) 
        pcder = tensor.sqrt(tensor.sqrt(v_t)+e) 
            
        eps = trng.normal(p.get_value().shape, avg = 0.0, std = 1.0, 
                          dtype=theano.config.floatX)
            
        m_t = -lr*g/pcder + (1. - alpha_t) * m + (tensor.sqrt(2*lr*v_t/(i_t ** anne_rate)/nframes) *eps) * tensor.lt(eidx, 0.15*max_epoch)
        p_t = p + (m_t/ pcder)
        
        updates.append((alpha, alpha_t))
        updates.append((m, m_t))
        updates.append((v, v_t))
        updates.append((p, p_t))
    updates.append((i, i_t))
    
    f_update = theano.function([lr,eidx,nframes,max_epoch], [], updates=updates)
    
    return f_grad_shared, f_update
    
