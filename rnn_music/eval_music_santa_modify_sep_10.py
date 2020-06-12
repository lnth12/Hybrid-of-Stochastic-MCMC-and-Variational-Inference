'''
Build an RNN Generative Model on the Polyphonic Music dataset
'''
import pdb
import sys
import time
import logging
#import cPickle
import scipy.io
import numpy as np
import theano
import theano.tensor as tensor
from collections import OrderedDict

from model.gru_model_modify_sep_10 import init_params_modify_sep_10, init_tparams, build_model_modify_sep_10
#from model.optimizers import Santa
from model.optimizers_modify_sep_10_v3 import Santa_modify, SGD
#from model.optimizers_modify_sep_10_v2 import Santa_modify

from theano.compile.nanguardmode import NanGuardMode

#theano.config.compute_test_value = 'off'

def v_dk(beta):
    betas = np.array([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    vs = np.array([-.33, -.472, -.631, -.792, -.953, -1.11, -1.29, -1.49, -1.74, -2.10, -10])
    from scipy.interpolate import interp1d
    return interp1d(betas, vs)(beta)

def zipp(params, tparams):
    """
    When we reload the model. Needed for the GPU stuff.
    """
    for kk, vv in params.iteritems():
        tparams[kk].set_value(vv)

def unzip(zipped):
    """
    When we pickle the model. Needed for the GPU stuff.
    """
    new_params = OrderedDict()
    for kk, vv in zipped.iteritems():
        new_params[kk] = vv.get_value()
    return new_params

def calc_negLoglike(f_cost, te):
    
    idx_list = np.arange(len(te), dtype="int32")
    total_cost = 0
    total_len = 0
    #pdb.set_trace()
    for idx in idx_list:
        x = te[idx] 
        n_steps = len(x)
        total_cost = total_cost + f_cost(x) * n_steps
        total_len = total_len + n_steps
       
    return total_cost / total_len
    
""" Training the model. """

def train_model(tr, va, te, n_x=88, n_h=200, patience=10, max_epochs=40, 
    lrate=0.001, ntrain = 10000, dispFreq=10, validFreq=200, 
    saveFreq=1000, saveto = 'example.npz', beta=0.5, batchsize=32):
        
    """ tr, va, te : datasets
        n_x : observation dimension
        n_h : LSTM/GRU number of hidden units 
        patience : Number of epoch to wait before early stop if no progress
        max_epochs : The maximum number of epoch to run
        lrate : learning rate
        ntrain : how any training time-steps we have in total
        dispFreq : Display to stdout the training progress every N updates
        validFreq : Compute the validation error after this number of update.
    """
    
    options = {}
    options['n_x'] = n_x
    options['n_h'] = n_h
    options['patience'] = patience
    options['max_epochs'] = max_epochs
    options['lrate'] = lrate
    options['dispFreq'] = dispFreq
    options['validFreq'] = validFreq
    options['saveFreq'] = saveFreq
   
    logger.info('Model options {}'.format(options))
    logger.info('Building model...')
    
    params = init_params_modify_sep_10(options, beta)
    tparams = init_tparams(params)

    (x, f_pred, cost) = build_model_modify_sep_10(tparams,options)
    
    #f_cost = theano.function([x], cost, name='f_cost', 
    #                         mode=NanGuardMode(nan_is_error=True, inf_is_error=True, big_is_error=True))
    #f_cost = theano.function([x], cost, name='f_cost', mode='DebugMode')
    f_cost = theano.function([x], cost, name='f_cost')
    lr_ = tensor.scalar(name='lr',dtype=theano.config.floatX)
    eidx_ = tensor.scalar(name='edix',dtype='int32')
    ntrain_ = tensor.scalar(name='ntrain',dtype='int32')
    maxepoch_ = tensor.scalar(name='maxepoch',dtype='int32')
    
    v_dk_beta = v_dk(beta)
    
    f_grad_shared, f_update = Santa_modify(beta, v_dk_beta, batchsize, tparams, cost, [x], lr_, eidx_, ntrain_, maxepoch_)
    #f_grad_shared, f_update = SGD(tparams, cost, [x], lrate)
    
    logger.info('Training model...')
    
    estop = False  # early stop
    history_negll = []
    best_p = None
    bad_counter = 0    
    uidx = 0  # the number of update done
    start_time = time.time()
    
    for eidx in xrange(max_epochs):
        idx_list = np.arange(len(tr), dtype="int32")
        #np.random.shuffle(idx_list)
        #print('epoch=', eidx)        
        #time.sleep(5)
        for train_index in idx_list:
            uidx += 1
            #print(uidx)
            #if uidx > 0:
            #pdb.set_trace()
            
            x = tr[train_index].astype(theano.config.floatX) 
            cost = f_grad_shared(x)
            #if tensor.isnan(cost):
            #if np.mod(uidx, 100) == 0:
            #    pdb.set_trace()
            #ddw_Lhat_wo_Ez, ddw_logprior_o_w_wq,  g, alpha, m, v, w, pcder = f_update(lrate,eidx,ntrain,max_epochs)
            ddw_Lhat_wo_Ez, ddw_logprior_o_w_wq,  g, alpha, v, w, pcder = f_update(lrate,eidx,ntrain,max_epochs)
            if np.mod(uidx, dispFreq) == 0:
                logger.info('Epoch {} Update {} Cost {} Beta {}'.format(eidx, uidx, cost, beta))
                
            if np.mod(uidx, saveFreq) == 0:
                logger.info('Saving ...')
                
                if best_p is not None:
                    params = best_p
                else:
                    params = unzip(tparams)
                np.savez(saveto, history_negll=history_negll, **params)
                logger.info('Done ...')

            if np.mod(uidx, validFreq) == 0:
                #pdb.set_trace()
                train_negll = calc_negLoglike(f_cost,tr)
                valid_negll = calc_negLoglike(f_cost,va)
                test_negll = calc_negLoglike(f_cost,te)
                history_negll.append([valid_negll, test_negll, train_negll])
                
                if (uidx == 0 or
                    valid_negll <= np.array(history_negll)[:,0].min()):
                        best_p = unzip(tparams)
                        bad_counter = 0
                        
                logger.info('Train {} Valid {} Test {} Beta {}'.format(train_negll, valid_negll, test_negll, beta))
                
                if (len(history_negll) > patience and
                     valid_negll >= np.array(history_negll)[:-patience,0].min()):
                        bad_counter += 1
                        if bad_counter > patience:
                            logger.info('Early Stop!')
                            estop = True
                            break
                        
        if estop:
            break

    end_time = time.time()
    if best_p is not None:
        zipp(best_p, tparams)
    else:
        best_p = unzip(tparams)
                
    train_negll = calc_negLoglike(f_cost,tr)
    valid_negll = calc_negLoglike(f_cost,va)
    test_negll = calc_negLoglike(f_cost,te)

    logger.info('Train {} Valid {} Test {} Beta {}'.format(train_negll, valid_negll, test_negll, beta))
    np.savez(saveto, train_negll=train_negll,
                valid_negll=valid_negll, test_negll=test_negll,
                history_negll=history_negll, **best_p)

    logger.info('The code run for {} epochs, with {} sec/epochs'.format(eidx + 1, 
                 (end_time - start_time) / (1. * (eidx + 1))))
    
    return train_negll, valid_negll, test_negll

if __name__ == '__main__':
    
    dataset = 'Piano'#sys.argv[1]
    lrate = 'small_lrate'#sys.argv[2]  
    
    # create logger with 'eval_music_santa'
    logger = logging.getLogger('eval_{}_santa'.format(dataset))
    logger.setLevel(logging.INFO)
    fh = logging.FileHandler('eval_{}_santa.log'.format(dataset))
    fh.setLevel(logging.INFO)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    logger.addHandler(fh)

    if dataset == "JSB":
        logger.info('loading JSB data...')
        data = scipy.io.loadmat('./data/JSB_Chorales.mat')
    elif dataset == "Muse":
        logger.info('loading Muse data...')
        data = scipy.io.loadmat('./data/MuseData.mat')
    elif dataset == "Nott":
        logger.info('loading Nott data...')
        data = scipy.io.loadmat('./data/Nottingham.mat')
    elif dataset == "Piano":
        logger.info('loading Piano data...')
        data = scipy.io.loadmat('./data/Piano_midi.mat')
    
    tr = data['traindata'][0]
    va = data['validdata'][0]
    te = data['testdata'][0]
    del data
    logger.info('data loaded!')
    
    #for beta in [0, 0.2, 0.4, 0.6, 0.8, 1.0]:    
    beta = 0.9
    
    saveto = "{}_Santa_with_Domke_beta_{}".format(dataset, beta)
    
    # validate the performance once every one epoch
    validFreq = len(tr)
    
    # calculate how many training frames we have in total
    ntrain = 0
    for seq in tr:
        ntrain = ntrain + len(seq)
        
    if lrate == "large_lrate":
        lrate_val = 1.*1e-3
    elif lrate == "small_lrate":
        lrate_val = 2.*1e-4
        
    [tr_negll, va_negll, te_negll] = train_model(tr, va, te, 
        lrate=lrate_val, ntrain = ntrain, validFreq=validFreq, 
        saveto = saveto, beta=beta)
