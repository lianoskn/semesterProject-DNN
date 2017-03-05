import configparser
import os

class Config:
    def __init__(self):
        pass
        
    def parseConfigFile(self, file):
           
        configFile = configparser.ConfigParser()

        configFile.read(file)
 
 
        """LOAD DIRECTORY INFO"""
        dir_info = configFile['DIRECTORY']
        self.root_dir = dir_info['root']
        self.traindata = dir_info['train_folder']
        self.testdata = dir_info['test_folder']
        self.isComposite = dir_info['isComposite']
        
        """CheckPoint INFO"""
        ckpt_info = configFile['CHECKPOINT']       
        self.checkpoint_dir = ckpt_info['checkpoint_dir']
        self.checkpoint_file = ckpt_info['checkpoint_file']
        self.SAVE_MODEL = (ckpt_info['save_model'])=='1'
        self.LOAD_RNN = (ckpt_info['load_rnn'])=='1'
        self.LOAD_CNN = (ckpt_info['load_cnn'])=='1'
        self.LOAD_CNN_checkpoint_dir = ckpt_info['saved_cnn_dir']
 
        """LOAD TRAINING INFO"""
        train_info = configFile['TRAINING']
        self.learning_rate = float(train_info['lr'])
        self.training_steps = float(train_info['training_steps'])
        self.n_classes = int(train_info['n_classes'])  
        self.dropout = float(train_info['dropout']) 
        self.display_step = int(train_info['display_step'])
        self.err_plot = train_info['error_plot_file']
        self.batchsize = int(train_info['batch_size']) 
        self.ignoresmall = int(train_info['ignoresmall']) 
        self.isTest = (train_info['testing'])=='1'
        self.logits_file = train_info['logits_file']
        """LOAD MODEL INFO"""
        model_info = configFile['MODELLING']
        self.IS_RNN = (model_info['isRNN'])=='1'
        self.IS_CNN = (model_info['isCNN'])=='1'
        
        model_info = configFile['RNN']
        self.nlayers = int(model_info['num_layers']) 
        self.rnnsize = int(model_info['rnn_size']) 
        self.rnn_input_dim = int(model_info['input_dim']) 
        self.maxseq = int(model_info['max_sequence_length']) 

        model_info = configFile['CNN']
        self.is_cnn_trainable = (model_info['is_trainable']) == '1'


        