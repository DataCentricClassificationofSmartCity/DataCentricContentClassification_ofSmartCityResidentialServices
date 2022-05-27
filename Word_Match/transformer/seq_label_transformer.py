from torch.optim.adam import Adam
from tqdm import tqdm

from src.training.transformer_model import Model
from src.training.transformer_layers import *
from src.training.transformer_utils import *
from word_match.transformer.decoder import Decoder


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
    tri_attn = get_subsequent_mask(input_sequence)

    return embedding_sequence, non_pad_mask, slf_attn_mask, tri_attn


def expand_3d_mask_to_n_head(input, n_head):
    input = input.unsqueeze(1).repeat(1, n_head, 1, 1)
    input = input.view(-1, input.shape[2], input.shape[3])
    return input


class SeqLableTransformer(Model):
    def __init__(
            self, save_path, log_path, n_layers, n_head, d_features, d_meta, enc_seq_length,
            dec_seq_length,
            d_k, d_v, d_bottleneck, n_vocab, threshold=None, embedding=None, init_lr=0.1, n_warmup_steps=4000,
            weight_decay=0):
        '''**kwargs: n_layers, n_head, dropout, use_bottleneck, d_bottleneck'''

        super().__init__(save_path, log_path)
        self.threshold = threshold
        self.dec_sec_length = dec_seq_length
        self.n_head = n_head
        self.n_vocab = n_vocab

        # ----------------------------- Model ------------------------------ #
        torch.manual_seed(0)

        self.encoder = Encoder(SinusoidPositionEncoding, n_layers=n_layers, n_groups=1, n_head=n_head,
                               d_features=d_features, max_seq_length=enc_seq_length,
                               d_meta=d_meta, d_k=d_k, d_v=d_v, d_bottleneck=d_bottleneck)

        self.decoder = Decoder(SinusoidPositionEncoding, n_layers=n_layers, n_groups=1, n_head=n_head,
                               d_features=d_features, max_seq_length=dec_seq_length,
                               d_meta=d_meta, d_k=d_k, d_v=d_v, d_bottleneck=d_bottleneck)

        # self.transformer = nn.Transformer(d_features, n_head, enc_layers, dec_layers, d_bottleneck)
        self.trg_word_prj = nn.Linear(d_features, n_vocab, bias=False)
        # --------------------------- Embedding  --------------------------- #
        # Placeholder: <pad>: 0, <ukn>: -3, <bos>: -2, <eos>: -1
        ph_embedding = torch.normal(0, 1, [4, 100], dtype=torch.float32)
        embedding = torch.FloatTensor(embedding)
        embedding = torch.cat([ph_embedding[:1], embedding, ph_embedding[1:]])
        self.vocab_embedding = nn.Embedding.from_pretrained(embedding, freeze=True, padding_idx=0)

        self.USE_EMBEDDING = True
        # ------------------------- Loss Function--------------------------- #
        # ------------------------------ CUDA ------------------------------ #
        self.data_parallel()

        # ---------------------------- Optimizer --------------------------- #
        self.parameters = list(self.encoder.parameters()) + list(self.decoder.parameters()) + list(
            self.trg_word_prj.parameters())
        # self.parameters = list(self.transformer.parameters())
        optimizer = Adam(self.parameters, betas=(0.9, 0.999), weight_decay=weight_decay)
        self.set_optimizer(optimizer, init_lr=init_lr, d_model=d_features, n_warmup_steps=n_warmup_steps)

        # ------------------------ training control ------------------------ #
        self.controller = TrainingControl(max_step=100000, evaluate_every_nstep=100, print_every_nstep=10)
        self.early_stopping = EarlyStopping(patience=50, mode='best', on='loss')

        # --------------------- logging and tensorboard -------------------- #
        self.set_logger()
        self.set_summary_writer()
        # ---------------------------- END INIT ---------------------------- #

    def data_parallel(self):
        # If GPU available, move the graph to GPU(s)
        self.CUDA_AVAILABLE = self.check_cuda()
        if self.CUDA_AVAILABLE:
            self.encoder.to('cuda')
            self.decoder.to('cuda')
            self.trg_word_prj.to('cuda')

        else:
            print('CUDA not found or not enabled, use CPU instead')

    def checkpoint(self, step):
        checkpoint = {
            'encoder_state_dict': self.encoder.state_dict(),
            'decoder_state_dict': self.decoder.state_dict(),
            'word_predictor': self.trg_word_prj.state_dict(),
            # 'model_state_dict': self.transformer.state_dict(),
            'embedding_dict': self.vocab_embedding.state_dict(),
            'optimizer_state_dict': self.optimizer._optimizer.state_dict(),
            'global_step': step}
        return checkpoint

    def loss(self, logits, target, ignore_index=0):
        loss_function = nn.CrossEntropyLoss(reduction='none', ignore_index=ignore_index)
        loss = loss_function(logits, target)
        return loss

    def train_epoch(self, train_dataloader, eval_dataloader, device, smoothing, earlystop):
        ''' Epoch operation in training phase'''

        if device == 'cuda':
            assert self.check_cuda()
        # Set model and classifier training mode

        total_loss = 0
        batch_counter = 0
        self.encoder.train()
        self.decoder.train()
        self.trg_word_prj.train()

        # update param per batch
        for batch in tqdm(train_dataloader, mininterval=1, desc='  - (Training)   ', leave=False):
            enc_input, enc_position, dec_input_indexes, dec_position, _ = map(lambda x: x.to(device), batch)
            batch_size = len(enc_input)
            enc_input, enc_non_pad_mask, enc_slf_attn_mask, _ = parse_data_enc(enc_input, self.vocab_embedding)
            dec_input, dec_non_pad_mask, dec_slf_attn_mask, dec_tri_mask = parse_data_enc(dec_input_indexes,
                                                                                          self.vocab_embedding)
            self.optimizer.zero_grad()
            enc_output, enc_slf_attn = self.encoder(enc_input, enc_position, enc_non_pad_mask, enc_slf_attn_mask)
            dec_output, dec_self_attn, dec_enc_attn = self.decoder(dec_input, dec_position, enc_output, dec_tri_mask)
            # logits = self.transformer(src=enc_input, tgt=dec_input, src_mask=enc_slf_attn_mask, tgt_mask=dec_tri_mask,
            #                           src_key_padding_mask=enc_non_pad_mask, tgt_key_padding_mask=dec_non_pad_mask)
            logits = self.trg_word_prj(dec_output)
            logits = logits.view(batch_size * self.dec_sec_length, -1)
            target = dec_input_indexes.clone().detach()

            # Shift to the left by 1 to get the target
            target[:, :-1] = dec_input_indexes[:, 1:]
            target = target.view(batch_size * self.dec_sec_length)
            loss = self.loss(logits, target, ignore_index=0)
            loss = loss.mean()

            # calculate gradients
            loss.backward()
            # ignore padding
            acc = accuracy(logits, target, ignore_idx=0)

            # update parameters
            self.optimizer.step_and_update_lr()

            # get metrics for logging
            total_loss += loss.item()
            batch_counter += 1

            # training control
            state_dict = self.controller(batch_counter)

            if state_dict['step_to_print']:
                self.train_logger.info(
                    '[TRAINING] - lr: %5f, step: %5d, loss: %3.4f, acc: %3f' %
                    (self.optimizer.get_current_lr(), state_dict['step'], loss, acc))
                self.summary_writer.add_scalar('loss/train', loss, state_dict['step'])

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
        self.encoder.eval()
        self.decoder.eval()
        self.trg_word_prj.eval()

        # use evaluator to calculate the average performance
        total_loss = 0

        with torch.no_grad():
            for batch in tqdm(dataloader, mininterval=5, desc='  - (Evaluation)   ', leave=False):
                enc_input, enc_position, dec_input_indexes, dec_position, _ = map(lambda x: x.to(device), batch)
                batch_size = len(enc_input)
                enc_input, enc_non_pad_mask, enc_slf_attn_mask, _ = parse_data_enc(enc_input, self.vocab_embedding)
                dec_input, dec_non_pad_mask, dec_slf_attn_mask, dec_tri_mask = parse_data_enc(dec_input_indexes,
                                                                                              self.vocab_embedding)
                enc_output, enc_slf_attn = self.encoder(enc_input, enc_position, enc_non_pad_mask, enc_slf_attn_mask)
                dec_output, dec_self_attn, dec_enc_attn = self.decoder(dec_input, dec_position, enc_output,
                                                                       dec_tri_mask)

                logits = self.trg_word_prj(dec_output)
                logits = logits.view(batch_size * self.dec_sec_length, -1)
                target = dec_input_indexes.clone().detach()

                # Shift to the left by 1 to get the target
                target[:, :-1] = dec_input_indexes[:, 1:]
                target = target.view(batch_size * self.dec_sec_length)
                loss = self.loss(logits, target, ignore_index=0)
                loss = loss.mean()
                # ignore padding
                acc = accuracy(logits, target, ignore_idx=0)
                total_loss += loss

        self.eval_logger.info(
            '[EVALUATING] - lr: %5f, step: %5d, loss: %3.4f' %
            (self.optimizer.get_current_lr(), step, total_loss))

        state_dict = self.early_stopping(total_loss.item())

        if state_dict['save']:
            checkpoint = self.checkpoint(step)
            self.save_model(checkpoint, self.save_path + '/model-step-%d-loss-%.5f' % (step, total_loss))

        return state_dict['break']

    def get_predictions(self, data_loader, device, output_length, bos_token=0, eos_token=0, greedy=True):

        if self.USE_EMBEDDING:
            self.vocab_embedding = self.vocab_embedding.to(device)

        self.encoder.eval()
        self.decoder.eval()
        self.trg_word_prj.eval()

        pred_list, real_list = [], []
        enc_slf_attn_list, dec_enc_attn_list = [], []

        with torch.no_grad():
            for batch in tqdm(data_loader, desc='  - (Test)   ', leave=None, position=0):
                sentence = []
                enc_input, enc_position, dec_input_indexes, dec_position, classes = map(lambda x: x.to(device), batch)
                enc_input, enc_non_pad_mask, enc_slf_attn_mask, _ = parse_data_enc(enc_input, self.vocab_embedding)
                dec_input, dec_non_pad_mask, dec_slf_attn_mask, dec_tri_mask = parse_data_enc(dec_input_indexes,
                                                                                              self.vocab_embedding)
                masked_input = dec_input_indexes.masked_fill(dec_tri_mask[0], 0)
                dec_position = dec_position.masked_fill(dec_tri_mask[0], 0)
                init_indexes = masked_input[0:1].clone().detach()
                for i in range(1, output_length):
                    if not greedy:
                        input_index = masked_input[i - 1:i]
                    else:
                        if i > 1:
                            input_index[0][i - 1] = word_idx[0]
                        else:
                            input_index = init_indexes

                    dec_position_ = dec_position[i - 1:i]
                    dec_input_, _, _, _ = parse_data_enc(input_index, self.vocab_embedding)

                    enc_output, enc_slf_attn = self.encoder(enc_input, enc_position, enc_non_pad_mask,
                                                            enc_slf_attn_mask)
                    enc_output, dec_self_attn, dec_enc_attn = self.decoder(dec_input_, dec_position_, enc_output,
                                                                           dec_tri_mask)
                    word_vector = enc_output[0][i - 1:i]
                    word = self.trg_word_prj(word_vector)
                    value, word_idx = F.softmax(word).topk(1)
                    if word_idx.item() == eos_token:
                        break
                    if word_idx.item() == bos_token:
                        continue
                    sentence.append(word_idx.item() - 1)
                pred_list.append(sentence)
                real_list.append(classes.item())
                enc_slf_attn_list.append(enc_slf_attn)
                dec_enc_attn_list.append(dec_enc_attn)

        return pred_list, real_list, enc_slf_attn_list, dec_enc_attn_list

    def train(self, max_epoch, train_dataloader, eval_dataloader, device, smoothing=False, earlystop=False,
              save_mode='best'):
        assert save_mode in ['all', 'best']

        if self.USE_EMBEDDING:
            self.vocab_embedding = self.vocab_embedding.to(device)

        # train for n epoch
        for epoch_i in range(max_epoch):
            print('[ Epoch', epoch_i, ']')
            # set current epoch
            self.controller.set_epoch(epoch_i + 1)
            # train for on epoch
            state_dict = self.train_epoch(train_dataloader, eval_dataloader, device, smoothing, earlystop)

        checkpoint = self.checkpoint(state_dict['step'])

        self.save_model(checkpoint, self.save_path + '/model-step-%d' % state_dict['step'])

        self.train_logger.info(
            '[INFO]: Finish Training, ends with %d epoch(s) and %d batches, in total %d training steps.' % (
                state_dict['epoch'] - 1, state_dict['batch'], state_dict['step']))

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

    def load_model(self, checkpoint_path, device):
        checkpoint = torch.load(checkpoint_path, map_location=torch.device(device))

        encoder_state_dict = checkpoint['encoder_state_dict']
        self.encoder.load_state_dict(encoder_state_dict)
        self.encoder.eval()

        decoder_state_dict = checkpoint['decoder_state_dict']
        self.decoder.load_state_dict(decoder_state_dict)
        self.decoder.eval()

        word_predictor_state_dict = checkpoint['word_predictor']
        self.trg_word_prj.load_state_dict(word_predictor_state_dict)
        self.trg_word_prj.eval()

        try:
            embedding_dict = checkpoint['embedding_dict']
            self.vocab_embedding.load_state_dict(embedding_dict)
            self.vocab_embedding.eval()
        except:
            print('[WARNING] Did not find a saved embedding')
