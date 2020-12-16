'''
@author Tian Shi
Please contact tshi@vt.edu
'''
import os
import time

import torch
from torch.autograd import Variable

from LeafNATS.modules.attention.nats_attention_encoder import AttentionEncoder
from LeafNATS.modules.encoder2decoder.nats_encoder2decoder import \
    natsEncoder2Decoder
from LeafNATS.modules.encoder.nats_encoder_rnn import natsEncoder
from UAE.model_NLG_base import modelNLGBase


class modelUAE(modelNLGBase):

    def __init__(self, args):
        super().__init__(args=args)

    def build_models(self):
        '''
        build all models.
        '''
        self.base_models['embedding'] = torch.nn.Embedding(
            self.batch_data['vocab_size'],
            self.args.emb_size
        ).to(self.args.device)

        self.train_models['gate'] = torch.nn.Linear(
            self.args.emb_size, self.args.emb_size
        ).to(self.args.device)
        self.train_models['value'] = torch.nn.Linear(
            self.args.emb_size, self.args.emb_size
        ).to(self.args.device)

        self.train_models['encoder'] = natsEncoder(
            emb_dim=self.args.emb_size,
            hidden_size=self.args.hidden_size,
            rnn_network='lstm',
            device=self.args.device
        ).to(self.args.device)

        self.train_models['aspect_embedding'] = torch.nn.Embedding(
            self.args.n_aspects,
            self.args.hidden_size
        ).to(self.args.device)

        self.train_models['aspect_ff'] = torch.nn.Linear(
            self.args.hidden_size,
            self.args.hidden_size*2
        ).to(self.args.device)

        self.train_models['encoder2decoder'] = natsEncoder2Decoder(
            src_hidden_size=self.args.hidden_size,
            trg_hidden_size=self.args.hidden_size,
            rnn_network='lstm',
            device=self.args.device
        ).to(self.args.device)

        self.train_models['decoderRNN'] = torch.nn.LSTMCell(
            self.args.emb_size+self.args.hidden_size,
            self.args.hidden_size
        ).to(self.args.device)

        self.train_models['attnEncoder0'] = AttentionEncoder(
            self.args.hidden_size,
            self.args.hidden_size,
            attn_method='luong_concat',
            repetition='vanilla'
        ).to(self.args.device)

        self.train_models['attnEncoder1'] = AttentionEncoder(
            self.args.hidden_size,
            self.args.hidden_size,
            attn_method='luong_concat',
            repetition='vanilla'
        ).to(self.args.device)

        self.train_models['wrapDecoder'] = torch.nn.Linear(
            self.args.hidden_size*3,
            self.args.hidden_size
        ).to(self.args.device)

        self.train_models['decoder2proj'] = torch.nn.Linear(
            self.args.hidden_size,
            self.batch_data['vocab_size']
        ).to(self.args.device)

        self.drop = torch.nn.Dropout(0.005).to(self.args.device)

    def build_pipe(self):
        '''
        data pipe
        '''
        bsize = self.batch_data['sen_var_input'].size(0)
        src_seq_len = self.batch_data['sen_var_output'].size(1)
        tgt_seq_len = self.batch_data['sen_var_input'].size(1)

        with torch.no_grad():
            src_emb = self.base_models['embedding'](
                self.batch_data['sen_var_output'])
            tgt_emb = self.base_models['embedding'](
                self.batch_data['sen_var_input'])
        # emb_gate = torch.sigmoid(self.train_models['gate'](src_emb))
        # emb_valu = torch.relu(self.train_models['value'](src_emb))
        # src_emb = src_emb*(1-emb_gate) + emb_valu*emb_gate
        # emb_gate = torch.sigmoid(self.train_models['gate'](tgt_emb))
        # emb_valu = torch.relu(self.train_models['value'](tgt_emb))
        # tgt_emb = tgt_emb*(1-emb_gate) + emb_valu*emb_gate
        src_enc, hidden_encoder = self.train_models['encoder'](src_emb)

        asp_emb = Variable(torch.LongTensor(
            range(self.args.n_aspects))).to(self.args.device)
        asp_emb = asp_emb.unsqueeze(0).repeat(bsize, 1)
        asp_emb = self.train_models['aspect_embedding'](asp_emb)
        # emb_gate = torch.sigmoid(self.train_models['gate'](asp_emb))
        # emb_valu = torch.relu(self.train_models['value'](asp_emb))
        # asp_emb = asp_emb*(1-emb_gate) + emb_valu*emb_gate

        if self.args.task == "train":
            mean = Variable(torch.zeros(
                asp_emb.size())).to(self.args.device)
            std = Variable(torch.ones(
                asp_emb.size())).to(self.args.device)
            var = torch.normal(mean, 0.1*std)
            asp_emb = asp_emb + var

        asp_emb = self.train_models['aspect_ff'](asp_emb)
        attn_aspect = asp_emb @ src_enc.transpose(1, 2)
        mask = self.batch_data['sen_pad_mask'].unsqueeze(1)
        mask = mask.repeat(1, self.args.n_aspects, 1)
        attn_aspect = attn_aspect.masked_fill(mask == 0, -1e20)
        # attn_aspect = self.drop(attn_aspect)
        # attn_aspect = attn_aspect.masked_fill(attn_aspect == 0, -1e20)
        attn_aspect = torch.softmax(attn_aspect, dim=2)
        ctx_aspect = attn_aspect @ src_enc

        h_attn = Variable(torch.zeros(
            bsize, self.args.hidden_size)).to(self.args.device)
        past_attn = Variable(torch.ones(
            bsize, self.args.n_aspects)/float(src_seq_len)
        ).to(self.args.device)

        output = []
        encoder_attn = []
        hidden = self.train_models['encoder2decoder'](hidden_encoder)
        for k in range(tgt_seq_len):
            dec_input = torch.cat((tgt_emb[:, k], h_attn), 1)
            hidden = self.train_models['decoderRNN'](dec_input, hidden)
            ctx_enc0, attn, _ = self.train_models['attnEncoder0'](
                hidden[0], ctx_aspect)
            ctx_enc1, _, _ = self.train_models['attnEncoder1'](
                hidden[0], src_enc, past_attn,
                self.batch_data['sen_pad_mask'])
            term_mask = self.batch_data['mask_terms'][:, k].unsqueeze(1)
            ctx_enc = term_mask*ctx_enc0 + (1.0-term_mask)*ctx_enc1
            encoder_attn.append(attn)
            h_attn = self.train_models['wrapDecoder'](
                torch.cat((ctx_enc, hidden[0]), 1))
            output.append(h_attn)
        tgt_out = torch.cat(output, 0)
        tgt_out = tgt_out.view(tgt_seq_len, bsize, -1)
        tgt_out = tgt_out.transpose(0, 1)
        encoder_attn = torch.cat(encoder_attn, 0)
        encoder_attn = encoder_attn.view(tgt_seq_len, bsize, -1)
        encoder_attn = encoder_attn.transpose(0, 1)
        logits = self.train_models['decoder2proj'](tgt_out)

        return logits, ctx_aspect, attn_aspect, encoder_attn
