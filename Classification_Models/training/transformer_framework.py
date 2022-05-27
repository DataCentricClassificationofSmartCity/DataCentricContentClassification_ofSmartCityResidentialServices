import logging
from abc import abstractmethod
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter


class ScheduledOptim:
    '''A simple wrapper class for learning rate scheduling'''
    def __init__(self, optimizer, init_lr, d_model, n_warmup_steps):
        """
        Args:
            optimizer:
            init_lr:
            d_model: d_word_vec
            n_warmup_steps:
        """
        self._optimizer = optimizer
        self.init_lr = init_lr
        self.current_lr = None
        self.d_model = d_model
        self.n_warmup_steps = n_warmup_steps
        self.n_steps = 0

    def step_and_update_lr(self):
        "Step with the inner optimizer"
        self._update_learning_rate()
        self._optimizer.step()

    def zero_grad(self):
        "Zero out the gradients with the inner optimizer"
        self._optimizer.zero_grad()

    def _get_lr_scale(self):
        d_model = self.d_model
        n_steps, n_warmup_steps = self.n_steps, self.n_warmup_steps
        return (d_model ** -0.5) * min(n_steps ** (-0.5), n_steps * n_warmup_steps ** (-1.5))

    def _update_learning_rate(self):
        ''' Learning rate scheduling per step '''
        self.n_steps += 1
        self.current_lr = self.init_lr * self._get_lr_scale()
        for param_group in self._optimizer.param_groups:
            param_group['lr'] = self.current_lr

    def get_current_lr(self):
        return self.current_lr


class Model:
    def __init__(self, save_path, log_path):
        self.save_path = save_path
        self.log_path = log_path
        self.model = None
        self.classifier = None
        self.parameters = None
        self.optimizer = None
        self.train_logger = None
        self.eval_logger = None
        self.summary_writer = None

    @abstractmethod
    def loss(self, **kwargs):
        pass

    @abstractmethod
    def train(self, **kwargs):
        pass

    @abstractmethod
    def evaluate(self, **kwargs):
        pass

    @abstractmethod
    def checkpoint(self, **kwargs):
        pass

    def data_parallel(self):
        # If GPU available, move the graph to GPU(s)
        self.CUDA_AVAILABLE = self.check_cuda()
        if self.CUDA_AVAILABLE:
            device_ids = list(range(torch.cuda.device_count()))
            self.model = nn.DataParallel(self.model, device_ids)
            self.classifier = nn.DataParallel(self.classifier, device_ids)
            self.model.to('cuda')
            self.classifier.to('cuda')
            assert next(self.model.parameters()).is_cuda
            assert next(self.classifier.parameters()).is_cuda
            pass

        else:
            print('CUDA not found or not enabled, use CPU instead')

    def set_optimizer(self, defined_optimizer, init_lr, d_model, n_warmup_steps):
        self.optimizer = ScheduledOptim(defined_optimizer, init_lr, d_model, n_warmup_steps)

    def set_logger(self, mode='a'):
        self.train_logger = logger('train', self.log_path + 'train_log', mode=mode)
        self.eval_logger = logger('eval', self.log_path + 'eval_log', mode=mode)

    def set_summary_writer(self):
        self.summary_writer = SummaryWriter(self.log_path + 'tensorboard')

    def check_cuda(self):
        if torch.cuda.is_available():
            print("INFO: CUDA device exists")
            return torch.cuda.is_available()

    def resume_checkpoint(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path)

        if self.model != None:
            model_state_dict = checkpoint['model_state_dict']
            self.model.load_state_dict(model_state_dict)
            self.model.train()

        classifier_state_dict = checkpoint['classifier_state_dict']
        self.classifier.load_state_dict(classifier_state_dict)
        self.classifier.train()

    def save_model(self, checkpoint, save_path):
        torch.save(checkpoint, save_path)

    def load_model(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path)

        if self.model != None:
            model_state_dict = checkpoint['model_state_dict']
            self.model.load_state_dict(model_state_dict)
            self.model.eval()

        classifier_state_dict = checkpoint['classifier_state_dict']
        self.classifier.load_state_dict(classifier_state_dict)
        self.classifier.eval()

    def count_parameters(self):
        try:
            assert self.model != None
            model_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            print('Number of Model Parameters: %d' % model_params)
            for name, param in self.model.named_parameters():
                if param.requires_grad:
                    print(name, param.numel())
        except:
            print('No Model specified')

        try:
            assert self.classifier != None
            classifier = sum(p.numel() for p in self.classifier.parameters() if p.requires_grad)
            print('Number of Model Classifier: %d' % classifier)
            for name, param in self.classifier.named_parameters():
                if param.requires_grad:
                    print(name, param.numel())
        except:
            print('No Classifier specified')


class logger():
    def __init__(self, logger_name, log_path, mode, level=logging.INFO, format="%(asctime)s - %(message)s"):
        self.logging = logging.getLogger(logger_name)
        self.logging.setLevel(level)
        fh = logging.FileHandler(log_path, mode)
        fh.setLevel(level)
        sh = logging.StreamHandler()
        sh.setLevel(logging.INFO)
        formatter = logging.Formatter(format)
        fh.setFormatter(formatter)
        sh.setFormatter(formatter)
        self.logging.addHandler(fh)
        self.logging.addHandler(sh)

    def info(self, msg):
        return self.logging.info(msg)
