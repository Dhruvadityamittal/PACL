from .base_new import *
import torch
from torchvision import transforms, datasets
from sklearn.preprocessing import LabelEncoder

class pamap(BaseDataset_new):
    def __init__(self, root, mode, windowlen,transform=None, autoencoderType = None,  standardize = None, fold = None):
        self.root = root
        self.mode = mode
        self.transform = transform
        self.autoencoderType = autoencoderType

        if(standardize):
            wln = str(windowlen) + '_standardized_True'
        else:
            wln = str(windowlen) + '_standardized_False'

        self.path_train_o_x = self.root + 'data_initial_step_scenario_1_x_train_windowLen_' + wln+'_fold'+str(fold) +'.npy'     
        self.path_train_o_y = self.root + 'data_initial_step_scenario_1_y_train_windowLen_' + wln+'_fold'+str(fold) +'.npy'
        self.path_train_o_p = self.root + 'data_initial_step_scenario_1_p_train_windowLen_' + wln+'_fold'+str(fold) +'.npy'   

        self.path_val_o_x = self.root + 'data_initial_step_scenario_1_x_val_windowLen_' + wln+'_fold'+str(fold) +'.npy'   
        self.path_val_o_y = self.root + 'data_initial_step_scenario_1_y_val_windowLen_' + wln+'_fold'+str(fold) +'.npy'
        self.path_val_o_p = self.root + 'data_initial_step_scenario_1_p_train_windowLen_' + wln+'_fold'+str(fold) +'.npy'    

        self.path_test_o_x = self.root + 'data_initial_step_scenario_1_x_test_windowLen_' + wln+'_fold'+str(fold) +'.npy'   
        self.path_test_o_y = self.root + 'data_initial_step_scenario_1_y_test_windowLen_' + wln+'_fold'+str(fold) +'.npy' 
        self.path_test_o_p = self.root + 'data_initial_step_scenario_1_p_test_windowLen_' + wln+'_fold'+str(fold) +'.npy'  

        self.path_train_n_1_x = self.root + 'data_incremental_step_scenario_1_x_train_windowLen_' + wln+'_fold'+str(fold) +'.npy'   
        self.path_train_n_1_y = self.root + 'data_incremental_step_scenario_1_y_train_windowLen_' + wln+'_fold'+str(fold) +'.npy'    
        self.path_train_n_1_p = self.root + 'data_incremental_step_scenario_1_p_train_windowLen_' + wln+'_fold'+str(fold) +'.npy' 

        self.path_val_n_1_x = self.root + 'data_incremental_step_scenario_1_x_val_windowLen_' + wln+'_fold'+str(fold) +'.npy'   
        self.path_val_n_1_y = self.root + 'data_incremental_step_scenario_1_y_val_windowLen_' + wln+'_fold'+str(fold) +'.npy'
        self.path_val_n_1_p = self.root + 'data_incremental_step_scenario_1_p_val_windowLen_' + wln+'_fold'+str(fold) +'.npy'   

        self.path_test_x = self.root + 'data_incremental_step_scenario_1_x_test_windowLen_' + wln+'_fold'+str(fold) +'.npy'   
        self.path_test_y = self.root + 'data_incremental_step_scenario_1_y_test_windowLen_' + wln+'_fold'+str(fold) +'.npy'  
        self.path_test_p = self.root + 'data_incremental_step_scenario_1_p_test_windowLen_' + wln+'_fold'+str(fold) +'.npy' 

        
        
        if self.mode == 'train_0':
            # self.classes = [ 0,  1,  2,  3,  4,  5,  6,  7, 11, 12, 13, 14, 18]
            self.classes = range(0,12)
            self.path_x = self.path_train_o_x
            self.path_y = self.path_train_o_y
            self.path_p = self.path_train_o_p

        elif self.mode == 'train_1':
            # self.classes = [ 1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16, 17, 18]
            self.classes = range(0,18)
            self.path_x = self.path_train_n_1_x
            self.path_y = self.path_train_n_1_y
            self.path_p = self.path_train_n_1_p

        elif self.mode == 'eval_0':
            self.classes = range(0,12)
            # self.classes = [ 0,  1,  2,  3,  4,  5,  6,  7, 11, 12, 13, 14, 18]
            self.path_x = self.path_val_o_x
            self.path_y = self.path_val_o_y
            self.path_p = self.path_val_o_p

        elif self.mode == 'eval_1':
            self.classes = range(0,18)
            # self.classes = [ 1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16, 17, 18]
            self.path_x = self.path_val_n_1_x
            self.path_y = self.path_val_n_1_y
            self.path_p = self.path_val_n_1_p

        elif self.mode == 'test_0':
            # self.classes =[ 0,  1,  2,  3,  4,  5,  6,  7, 11, 12, 13, 14, 18]
            self.classes = range(0,12)
            self.path_x = self.path_test_o_x
            self.path_y = self.path_test_o_y
            self.path_p = self.path_test_o_p

        elif self.mode == 'test_1':
            
            # self.classes = [ 1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16, 17, 18]
            self.classes = range(0,18)
            self.path_x = self.path_test_x
            self.path_y = self.path_test_y
            self.path_p = self.path_test_p

        
        BaseDataset_new.__init__(self, self.path_x,self.path_y, self.mode, self.transform)

        
        
        index = 0
        for x,y,p in zip(np.load(self.path_x),np.load(self.path_y), np.load(self.path_p)):            
            # self.xs += [x]
            self.ys += [y]
            self.xs += [x]
            self.ps += [p] 
            self.I += [index]
            index += 1
        
