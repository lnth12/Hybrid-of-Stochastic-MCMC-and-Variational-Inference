# -*- coding: utf-8 -*-
"""
Created on Mon Oct 05 14:09:57 2015

@author: Zhe Gan
"""

import numpy as np
import matplotlib.pyplot as plt
plt.style.use('ggplot')

def adjustFigAspect(fig,aspect=1):
    '''
    Adjust the subplot parameters so that the figure has the correct
    aspect ratio.
    '''
    xsize,ysize = fig.get_size_inches()
    minsize = min(xsize,ysize)
    xlim = .4*minsize/xsize
    ylim = .4*minsize/ysize
    if aspect < 1:
        xlim *= aspect
    else:
        ylim /= aspect
    fig.subplots_adjust(left=.5-xlim,
                        right=.5+xlim,
                        bottom=.5-ylim,
                        top=.5+ylim)

""" Piano dataset. """

data_07 = np.load('Piano_Santa_with_Domke_beta_0.7.npz')
santa_domke_train_negll_07 = data_07['train_negll']
santa_domke_valid_negll_07 = data_07['valid_negll']
santa_domke_test_negll_07 = data_07['test_negll']
santa_domke_history_negll_07 = data_07['history_negll']

data_08 = np.load('Piano_Santa_with_Domke_beta_0.8.npz')
santa_domke_train_negll_08 = data_08['train_negll']
santa_domke_valid_negll_08 = data_08['valid_negll']
santa_domke_test_negll_08 = data_08['test_negll']
santa_domke_history_negll_08 = data_08['history_negll']

data_09 = np.load('Piano_Santa_with_Domke_beta_0.9.npz')
santa_domke_train_negll_09 = data_09['train_negll']
santa_domke_valid_negll_09 = data_09['valid_negll']
santa_domke_test_negll_09 = data_09['test_negll']
santa_domke_history_negll_09 = data_09['history_negll']

data_10 = np.load('Piano_Santa_with_Domke_beta_1.0.npz')
santa_domke_train_negll_10 = data_10['train_negll']
santa_domke_valid_negll_10 = data_10['valid_negll']
santa_domke_test_negll_10 = data_10['test_negll']
santa_domke_history_negll_10 = data_10['history_negll']

#data = np.load('piano_santa_explore_refine.npz')
#santa_train_negll = data['train_negll']
#santa_valid_negll = data['valid_negll']
#santa_test_negll = data['test_negll']
#santa_history_negll = data['history_negll']

#data = np.load('piano_santa_small_learn_rate.npz')
#santa_s_train_negll = data['train_negll']
#santa_s_valid_negll = data['valid_negll']
#santa_s_test_negll = data['test_negll']
#santa_s_history_negll = data['history_negll']

fig = plt.figure()
adjustFigAspect(fig,aspect=1.2)
ax = fig.add_subplot(111)
plt.plot(santa_domke_history_negll_07[:,2],'c', label='beta=0.7',linewidth=2.0, alpha=1.)
plt.plot(santa_domke_history_negll_08[:,2],'g', label='beta=0.8',linewidth=2.0, alpha=1.)
plt.plot(santa_domke_history_negll_09[:,2],'m', label='beta=0.9',linewidth=2.0, alpha=1.)
plt.plot(santa_domke_history_negll_10[:,2],'r', label='beta=1.0 (Santa orginal)',linewidth=2.0, alpha=1.)

#plt.plot(santa_history_negll[:,2],'k', label='Santa',linewidth=2.0, alpha=1.)
plt.legend(prop={'size':12})
plt.ylim((6,65))
plt.xlim((0,30))
plt.xlabel('Epochs',fontsize=12)
plt.ylabel('Negative Log-likelihood',fontsize=12)
plt.title('Piano',fontsize=12)
plt.savefig('piano_train_with_title.pdf',bbox_inches = 'tight')

'''
fig = plt.figure()
adjustFigAspect(fig,aspect=1.2)
ax = fig.add_subplot(111)
plt.plot(santa_domke_history_negll[:,0],'c', label='beta=0.5',linewidth=2.0, alpha=1.)
plt.plot(santa_history_negll[:,0],'k', label='Santa',linewidth=2.0, alpha=1.)
plt.legend(prop={'size':12})
plt.ylim((6,65))
plt.xlim((0,30))
plt.xlabel('Epochs',fontsize=12)
plt.ylabel('Negative Log-likelihood',fontsize=12)
plt.title('Piano valid',fontsize=12)
plt.savefig('piano_valid_with_title.pdf',bbox_inches = 'tight')


fig = plt.figure()
adjustFigAspect(fig,aspect=1.2)
ax = fig.add_subplot(111)
plt.plot(santa_domke_history_negll[:,1],'c', label='beta=0.5',linewidth=2.0, alpha=1.)
plt.plot(santa_history_negll[:,1],'k', label='Santa',linewidth=2.0, alpha=1.)
plt.legend(prop={'size':12})
plt.ylim((6,65))
plt.xlim((0,30))
plt.xlabel('Epochs',fontsize=12)
plt.ylabel('Negative Log-likelihood',fontsize=12)
plt.title('Piano test',fontsize=12)
plt.savefig('piano_test_with_title.pdf',bbox_inches = 'tight')
'''
