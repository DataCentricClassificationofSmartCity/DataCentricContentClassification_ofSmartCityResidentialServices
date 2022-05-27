from torch.optim.adam import Adam
import torch.nn as nn
from tqdm import tqdm

from src.training.transformer_framework import Model
from src.training.transformer_layers import *
from src.training.transformer_utils import *


def parse_data_enc(input_sequence, embedding):
    '''
    Returns:
        enc_output {Tensor, [batch_size, seq_length, d_v]} --
        non_pad_mask {Tensor, [n_head, seq_length, 1]} --
        slf_attn_mask {Tensor, [batch_size, seq_length, seq_length]} --
    '''

    slf_attn_mask = get_attn_key_pad_mask(seq_k=input_sequence, seq_q=input_sequence, padding_idx=0)
    non_pad_mask = get_non_pad_mask(input_sequence, padding_idx=0)

    embedding_sequence = embedding(input_sequence)

    return embedding_sequence, non_pad_mask, slf_attn_mask


class TransformerClassifierModel(Model):
    def __init__(
            self, save_path, log_path, d_features, d_meta, max_length, d_classifier, n_classes, threshold=None,
            embedding=None, stack='Encoder', position_encode='SinusoidPositionEncoding',
            init_lr=0.1, n_warmup_steps=4000, weight_decay=0, monitor_on='acc', **kwargs):
        '''**kwargs: n_layers, n_head, dropout, use_bottleneck, d_bottleneck'''

        super().__init__(save_path, log_path)
        self.d_output = n_classes
        self.threshold = threshold
        self.max_length = max_length

        # ----------------------------- Model ------------------------------ #
        stack_dict = {
            'Encoder': Encoder,
        }
        encoding_dict = {
            'SinusoidPositionEncoding': SinusoidPositionEncoding,
            'LinearPositionEncoding': LinearPositionEncoding,
            'TimeFacilityEncoding': TimeFacilityEncoding
        }

        self.model = stack_dict[stack](encoding_dict[position_encode], d_features=d_features, max_seq_length=max_length,
                                       d_meta=d_meta, **kwargs)

        # --------------------------- Embedding  --------------------------- #
        if len(embedding) == 0:
            self.word_embedding = None
            self.USE_EMBEDDING = False

        else:
            self.word_embedding = nn.Embedding.from_pretrained(embedding)
            self.USE_EMBEDDING = True

        # --------------------------- Classifier --------------------------- #
        self.classifier = LinearClassifier(d_features * max_length, d_classifier, n_classes)

        # ------------------------------ CUDA ------------------------------ #
        self.data_parallel()

        # ---------------------------- Optimizer --------------------------- #
        self.parameters = list(self.model.parameters()) + list(self.classifier.parameters())
        optimizer = Adam(self.parameters, betas=(0.9, 0.999), weight_decay=weight_decay)
        self.set_optimizer(optimizer, init_lr=init_lr, d_model=d_features, n_warmup_steps=n_warmup_steps)

        # ------------------------ training control ------------------------ #
        self.controller = TrainingControl(max_step=100000, evaluate_every_nstep=100, print_every_nstep=10)
        self.early_stopping = EarlyStopping(patience=50, on=monitor_on)

        # --------------------- logging and tensorboard -------------------- #
        self.set_logger()
        self.set_summary_writer()
        # ---------------------------- END INIT ---------------------------- #

    def checkpoint(self, step):
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'classifier_state_dict': self.classifier.state_dict(),
            'optimizer_state_dict': self.optimizer._optimizer.state_dict(),
            'global_step': step}
        return checkpoint

    def train_epoch(self, train_dataloader, eval_dataloader, device, smoothing, earlystop):
        ''' Epoch operation in training phase'''

        if device == 'cuda':
            assert self.CUDA_AVAILABLE
        # Set model and classifier training mode

        total_loss = 0
        batch_counter = 0

        # update param per batch
        for batch in tqdm(train_dataloader, mininterval=10, desc='  - (Training)   ', leave=True, position=0):

            self.model.train()
            self.classifier.train()

            # get data from dataloader
            index, position, y = map(lambda x: x.to(device), batch)
            batch_size = len(index)
            input_feature_sequence, non_pad_mask, slf_attn_mask = parse_data_enc(index, self.word_embedding)

            # forward
            self.optimizer.zero_grad()
            logits, attn = self.model(input_feature_sequence, position, non_pad_mask, slf_attn_mask)

            logits = logits.view(batch_size, -1)
            logits = self.classifier(logits)

            # Judge if it's a regression problem
            if self.d_output == 1:
                pred = logits.sigmoid()
                loss = mse_loss(pred, y)
            else:
                pred = logits
                loss = cross_entropy_loss(pred, y, smoothing=smoothing)

            # calculate gradients
            loss.backward()

            # update parameters
            self.optimizer.step_and_update_lr()

            # get metrics for logging
            acc = accuracy(pred, y, threshold=self.threshold)
            _, _, precision, recall = precision_recall(pred, y, self.d_output, threshold=self.threshold)
            total_loss += loss.item()
            batch_counter += 1

            # training control
            state_dict = self.controller(batch_counter)

            if state_dict['step_to_print']:
                self.train_logger.info(
                    '[TRAINING] - lr: %5f, step: %5d, loss: %3.4f, acc: %1.4f, pre: %1.4f, rec: %1.4f' %
                    (self.optimizer.get_current_lr(), state_dict['step'], loss, acc, precision, recall))
                self.summary_writer.add_scalar('loss/train', loss, state_dict['step'])
                self.summary_writer.add_scalar('acc/train', acc, state_dict['step'])
                self.summary_writer.add_scalar('precision/train', precision, state_dict['step'])
                self.summary_writer.add_scalar('recall/train', recall, state_dict['step'])

            if state_dict['step_to_evaluate']:
                stop = self.val_epoch(eval_dataloader, device, state_dict['step'])
                state_dict['step_to_stop'] = stop

                if earlystop & stop:
                    break

            if self.controller.current_step == self.controller.max_step:
                state_dict['step_to_stop'] = True
                break

        return state_dict

    def val_epoch(self, dataloader, device, step=0):
        ''' Epoch operation in evaluation phase '''
        if device == 'cuda':
            assert self.CUDA_AVAILABLE

        # Set model and classifier training mode
        self.model.eval()
        self.classifier.eval()

        # use evaluator to calculate the average performance
        evaluator = Evaluator()

        with torch.no_grad():
            for batch in tqdm(dataloader, mininterval=10, desc='  - (Evaluation)   ', leave=True, position=1):
                index, position, y = map(lambda x: x.to(device), batch)
                batch_size = len(index)
                input_feature_sequence, non_pad_mask, slf_attn_mask = parse_data_enc(index, self.word_embedding)

                # get logits
                logits, attn = self.model(input_feature_sequence, position, non_pad_mask, slf_attn_mask)
                logits = logits.view(batch_size, -1)
                logits = self.classifier(logits)

                if self.d_output == 1:
                    pred = logits.sigmoid()
                    loss = mse_loss(pred, y)
                else:
                    pred = logits
                    loss = cross_entropy_loss(pred, y, smoothing=False)

                acc = accuracy(pred, y, threshold=self.threshold)
                _, _, precision, recall = precision_recall(pred, y, self.d_output, threshold=self.threshold)

                # feed the metrics in the evaluator
                evaluator(loss.item(), acc.item(), precision.item(), recall.item())

            # get evaluation results from the evaluator
            loss_avg, acc_avg, pre_avg, rec_avg = evaluator.avg_results()

            self.eval_logger.info(
                '[EVALUATING] - lr: %5f, step: %5d, loss: %3.4f, acc: %1.4f, pre: %1.4f, rec: %1.4f' %
                (self.optimizer.get_current_lr(), step, loss_avg, acc_avg, pre_avg, rec_avg))
            self.summary_writer.add_scalar('loss/eval', loss_avg, step)
            self.summary_writer.add_scalar('acc/eval', acc_avg, step)
            self.summary_writer.add_scalar('precision/eval', pre_avg, step)
            self.summary_writer.add_scalar('recall/eval', rec_avg, step)

            state_dict = self.early_stopping(loss_avg)

            if state_dict['save']:
                checkpoint = self.checkpoint(step)
                self.save_model(checkpoint, self.save_path + 'model-step-%d_loss-%.5f' % (step, loss_avg))

            return state_dict['break']

    def train(self, max_epoch, train_dataloader, eval_dataloader, device, smoothing=False, earlystop=False,
              save_mode='best'):
        assert save_mode in ['all', 'best']

        if self.USE_EMBEDDING:
            self.word_embedding = self.word_embedding.to(device)

        # train for n epoch
        for epoch_i in range(max_epoch):
            print('[ Epoch', epoch_i, ']')
            # set current epoch
            self.controller.set_epoch(epoch_i + 1)
            # train for on epoch
            state_dict = self.train_epoch(train_dataloader, eval_dataloader, device, smoothing, earlystop)

        checkpoint = self.checkpoint(state_dict['step'])

        self.save_model(checkpoint, self.save_path + 'model-step-%d' % state_dict['step'])

        self.train_logger.info(
            '[INFO]: Finish Training, ends with %d epoch(s) and %d batches, in total %d training steps.' % (
                state_dict['epoch'] - 1, state_dict['batch'], state_dict['step']))

    def get_predictions(self, data_loader, device, max_batches=None, activation=None):

        if self.USE_EMBEDDING:
            self.word_embedding = self.word_embedding.to(device)

        pred_list, real_list = [], []

        self.model.eval()
        self.classifier.eval()

        batch_counter = 0

        with torch.no_grad():
            for batch in tqdm(data_loader, desc='  - (Testing)   ', leave=True):
                index, position, y = map(lambda x: x.to(device), batch)
                input_feature_sequence, non_pad_mask, slf_attn_mask = parse_data_enc(index, self.word_embedding)

                # get logits
                logits, attn = self.model(input_feature_sequence, position, non_pad_mask, slf_attn_mask)
                logits = logits.view(logits.shape[0], -1)
                logits = self.classifier(logits)

                # Whether to apply activation function
                if activation is not None:
                    pred = activation(logits)
                else:
                    pred = logits.softmax(dim=-1)
                pred_list += pred.tolist()
                real_list += y.tolist()

                if max_batches is not None:
                    batch_counter += 1
                    if batch_counter >= max_batches:
                        break

        return pred_list, real_list

    def get_prediction_proba_and_attn_output(self, input_feature_sequence, position, non_pad_mask, slf_attn_mask, y):
        """
        How to get inputs: 👇
        index, position, y = map(lambda x: x.to('cuda'), next(dataloader.test_loader.__iter__()))
        input_feature_sequence, non_pad_mask, slf_attn_mask = parse_data_enc(index, self.word_embedding)
        """
        # get logits
        logits, attn = self.model(input_feature_sequence, position, non_pad_mask, slf_attn_mask)
        logits = logits.view(logits.shape[0], -1)
        logits = self.classifier(logits)

        # Whether to apply activation function
        pred = logits.softmax(dim=-1)

        return pred, y, attn
