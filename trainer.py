import shutil
import keras
from keras.callbacks import ModelCheckpoint
from utils import * 

class Trainer():
    def __init__(self, config, model, X_train, y_train, X_val, y_val, optimizer = 'adam', loss_fn = 'mse', metrics = 'acc'):
        #self.config = config
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.monitor = 'val_{}'.format(metrics)
        if metrics == 'f1_score':
            metrics_fn = f1_score
            self.monitor_mode = 'max'
        elif metrics == 'acc':
            metrics_fn = 'acc'
            self.monitor_mode = 'max'
        elif metrics == 'mse':
            metrics_fn = 'mse'
            self.monitor = 'val_mean_squared_error'
            self.monitor_mode = 'min'
        else:
            metrics = 'acc'
            metrics_fn = 'acc'
            self.monitor_mode = 'max'
        self.model.compile(optimizer = optimizer, loss = loss_fn, metrics = [metrics_fn])
        # data
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val
        # trainer attributes
        self.name = config.model_name
        self.total_epoch = 0
        self.history = []
        if config.load:
            self.load()
        #self.best_f1 = 0
        self.batch_size = config.batch_size
    #
    #
    def train(self, epoch = 1, verbose = True):
        filepath = 'models/{}.checkpoint.pth.h5'.format(self.name)
        checkpoint = ModelCheckpoint(filepath, monitor=self.monitor, verbose=1, 
                    save_best_only=True, save_weights_only=True, mode = self.monitor_mode)
        #best_checkpoint = ModelCheckpoint(filepath, monitor='f1', verbose=1, save_best_only=True, mode='max')
        callbacks_list = [checkpoint]#, best_checkpoint]
        hist = self.model.fit(self.X_train, self.y_train, validation_data = (self.X_val, self.y_val), epochs = epoch, 
                        batch_size = self.batch_size, verbose = verbose, callbacks = callbacks_list) 
        self.history.append( hist.history )
        self.save()
    #====================================================
    # Trainer save/load
    #====================================================
    def save(self):
        params = {'total_epoch': self.total_epoch,
                  'batch_size':self.batch_size,
                  'history':self.history }
        dump(params, 'models/{}.checkpoint.pth.params'.format(self.name))
    def load(self):
        params = load('models/{}.checkpoint.pth.params'.format(self.name))
        self.total_epoch = params['total_epoch']
        self.batch_size = params['batch_size']
        self.history = params['history']
        self.model = keras.models.load_weights('models/{}.checkpoint.pth.h5'.format(self.name))





