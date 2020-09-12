# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import pdb
import torch
import torch.nn as nn
from transformers import RobertaConfig, RobertaForMaskedLM
from empchat.datasets.tokens import (
    BERT_ID,
    EMPTYPERSONA_TOKEN,
    PAD_TOKEN,
    PARLAI_PAD_TOKEN,
    START_OF_COMMENT,
)

class BertWrapper(torch.nn.Module):
    """
    Adds a optional transformer layer and a linear layer on top of BERT.
    """

    def __init__(
        self,
        bert_model,
        output_dim,
        add_transformer_layer=False,
        layer_pulled=-1,
        aggregation="first",
    ):
        # pdb.set_trace()
        super(BertWrapper, self).__init__()
        self.layer_pulled = layer_pulled
        self.aggregation = aggregation
        self.add_transformer_layer = add_transformer_layer
        
        # deduce bert output dim from the size of embeddings
        bert_output_dim = bert_model.roberta.embeddings.word_embeddings.weight.size(1)

        # if add_transformer_layer:
        #     config_for_one_layer = BertConfig(
        #         0,
        #         hidden_size=bert_output_dim,
        #         num_attention_heads=int(bert_output_dim / 64),
        #         intermediate_size=3072,
        #         hidden_act='gelu',
        #     )
        #     self.additional_transformer_layer = BertLayer(config_for_one_layer)
        self.additional_linear_layer = torch.nn.Linear(bert_output_dim, output_dim)
        self.bert_model = bert_model

    def forward(self, token_ids, segment_ids, attention_mask):
        """
        Forward pass.
        """
        
        # output_bert, output_pooler = self.bert_model(
        #     input_ids=token_ids,
        #     attention_mask=attention_mask,
        #     token_type_ids=segment_ids
        # )
        _, output_bert = self.bert_model(
            input_ids=token_ids,
            attention_mask=attention_mask,
            token_type_ids=segment_ids,
            output_hidden_states=True
        )
       
        # pdb.set_trace()
        # output_bert is a list of 12 (for bert base) layers.
        # if use bert_pretrained_model library:
        layer_of_interest = output_bert[self.layer_pulled]
        
        # else (use transformers lib):
        # layer_of_interest = output_bert
        
        dtype = next(self.parameters()).dtype
        if self.add_transformer_layer:
            # Follow up by yet another transformer layer
            extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
            extended_attention_mask = (~extended_attention_mask).to(dtype) * neginf(
                dtype
            )
            embedding_layer = self.additional_transformer_layer(
                layer_of_interest, extended_attention_mask
            )
        else:
            embedding_layer = layer_of_interest

        if self.aggregation == "mean":
            #  consider the average of all the output except CLS.
            # obviously ignores masked elements
            outputs_of_interest = embedding_layer[:, 1:, :]
            mask = attention_mask[:, 1:].type_as(embedding_layer).unsqueeze(2)
            sumed_embeddings = torch.sum(outputs_of_interest * mask, dim=1)
            nb_elems = torch.sum(attention_mask[:, 1:].type(dtype), dim=1).unsqueeze(1)
            embeddings = sumed_embeddings / nb_elems
        elif self.aggregation == "max":
            #  consider the max of all the output except CLS
            outputs_of_interest = embedding_layer[:, 1:, :]
            mask = (~attention_mask[:, 1:]).type(dtype).unsqueeze(2) * neginf(dtype)
            embeddings, _ = torch.max(outputs_of_interest + mask, dim=1)
        else:
            # easiest, we consider the output of "CLS" as the embedding
             embeddings = embedding_layer[:, 0, :]
        
        # We need this in case of dimensionality reduction
        result = self.additional_linear_layer(embeddings)
        
        # Sort of hack to make it work with distributed: this way the pooler layer
        # is used for grad computation, even though it does not change anything...
        # in practice, it just adds a very (768*768) x (768*batchsize) matmul
        # result += 0 * torch.sum(output_pooler)
        return result


class BertAdapter(nn.Module):
    def __init__(self, opt, dictionary):
        super().__init__()
        config = RobertaConfig.from_pretrained(
            "/content/drive/My Drive/PhoBERT_EMPATHETICDIALOGUES/EmpatheticDialogues/PhoBert/PhoBERT_base_transformers/config.json"
        )
        self.opt = opt
        self.pad_idx = dictionary[PAD_TOKEN]
        self.ctx_bert = BertWrapper(
            bert_model=RobertaForMaskedLM.from_pretrained(
                "/content/drive/My Drive/PhoBERT_EMPATHETICDIALOGUES/EmpatheticDialogues/PhoBert/PhoBERT_base_transformers/model.bin",
                config=config
            ),
            output_dim=opt.bert_dim,
            add_transformer_layer=opt.bert_add_transformer_layer,
        )
        self.cand_bert = BertWrapper(
            bert_model=RobertaForMaskedLM.from_pretrained(
                "/content/drive/My Drive/PhoBERT_EMPATHETICDIALOGUES/EmpatheticDialogues/PhoBert/PhoBERT_base_transformers/model.bin",
                config=config
            ),
            output_dim=opt.bert_dim,
            add_transformer_layer=opt.bert_add_transformer_layer,
        )

        # Reset the embeddings for the until-now unused BERT tokens
        orig_embedding_weights = RobertaForMaskedLM.from_pretrained(
            "/content/drive/My Drive/PhoBERT_EMPATHETICDIALOGUES/EmpatheticDialogues/PhoBert/PhoBERT_base_transformers/model.bin",
            config=config
        ).roberta.embeddings.word_embeddings.weight
        mean_val = orig_embedding_weights.mean().item()
        std_val = orig_embedding_weights.std().item()
        unused_tokens = [START_OF_COMMENT, PARLAI_PAD_TOKEN, EMPTYPERSONA_TOKEN]
        unused_token_idxes = [dictionary[token] for token in unused_tokens]
        for token_idx in unused_token_idxes:
            rand_embedding = orig_embedding_weights.new_empty(
                (1, orig_embedding_weights.size(1))
            ).normal_(mean=mean_val, std=std_val)
            for embeddings in [
                self.ctx_bert.bert_model.roberta.embeddings.word_embeddings,
                self.cand_bert.bert_model.roberta.embeddings.word_embeddings,
            ]:
                embeddings.weight[token_idx] = rand_embedding
        self.ctx_bert.bert_model.roberta.embeddings.word_embeddings.weight.detach_()
        self.cand_bert.bert_model.roberta.embeddings.word_embeddings.weight.detach_()

    def forward(self, context_w, cands_w):
        if context_w is not None:
            context_segments = torch.zeros_like(context_w)
            context_mask = (context_w != self.pad_idx).long()
            context_h = self.ctx_bert(
                token_ids=context_w,
                segment_ids=context_segments,
                attention_mask=context_mask,
            )
            if self.opt.normalize_sent_emb:
                context_h = context_h / context_h.norm(2, dim=1, keepdim=True)
        else:
            context_h = None
        if cands_w is not None:
            cands_segments = torch.zeros_like(cands_w)
            cands_mask = (cands_w != self.pad_idx).long()
            cands_h = self.cand_bert(
                token_ids=cands_w, 
                segment_ids=cands_segments, 
                attention_mask=cands_mask
            )
            if self.opt.normalize_sent_emb:
                cands_h = cands_h / cands_h.norm(2, dim=1, keepdim=True)
        else:
            cands_h = None
        return context_h, cands_h
