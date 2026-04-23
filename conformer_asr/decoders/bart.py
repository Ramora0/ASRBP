"""BART-family decoder used as a standalone ``*ForCausalLM`` with cross-attn.

``_CompatBartForCausalLM`` works around a long-standing interaction bug
between ``SpeechEncoderDecoderModel`` and ``BartForCausalLM``: the outer model
passes ``input_ids``, ``BartForCausalLM.forward`` internally also builds
``inputs_embeds`` and forwards both down to ``BartDecoder``, which rejects the
combination.

We bypass ``BartForCausalLM.forward`` entirely: pre-embed the ids ourselves
and pass only ``inputs_embeds`` into the inner decoder, then apply the LM
head. Loss is (re)computed here if ``labels`` is given, mirroring the parent's
behavior.
"""
from __future__ import annotations

from torch.nn import CrossEntropyLoss
from transformers import BartConfig, BartForCausalLM
from transformers.modeling_outputs import CausalLMOutputWithCrossAttentions


class _CompatBartForCausalLM(BartForCausalLM):
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        head_mask=None,
        cross_attn_head_mask=None,
        past_key_values=None,
        inputs_embeds=None,
        labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        cache_position=None,
        **kwargs,
    ):
        if input_ids is not None:
            bart_decoder = self.model.decoder
            embed_scale = getattr(bart_decoder, "embed_scale", 1.0)
            inputs_embeds = bart_decoder.embed_tokens(input_ids) * embed_scale
            input_ids = None

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.model.decoder(
            input_ids=None,
            attention_mask=attention_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            head_mask=head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
            cache_position=cache_position,
        )

        logits = self.lm_head(outputs[0])

        loss = None
        if labels is not None:
            labels = labels.to(logits.device)
            loss = CrossEntropyLoss()(
                logits.view(-1, self.config.vocab_size),
                labels.view(-1),
            )

        if not return_dict:
            output = (logits,) + tuple(
                v for v in (
                    outputs.past_key_values,
                    outputs.hidden_states,
                    outputs.attentions,
                    outputs.cross_attentions,
                )
                if v is not None
            )
            return ((loss,) + output) if loss is not None else output

        return CausalLMOutputWithCrossAttentions(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            cross_attentions=outputs.cross_attentions,
        )


def build_bart_decoder(
    mcfg,
    *,
    vocab_size: int,
    pad_id: int,
    bos_id: int,
    eos_id: int,
) -> _CompatBartForCausalLM:
    dec_cfg = BartConfig(
        vocab_size=vocab_size,
        d_model=mcfg.decoder_d_model,
        decoder_layers=mcfg.decoder_layers,
        decoder_attention_heads=mcfg.decoder_attention_heads,
        decoder_ffn_dim=mcfg.decoder_ffn_dim,
        # BartForCausalLM still reads encoder_layers/heads for self-attention
        # inside BartDecoder when used standalone; set them to match the
        # decoder shape so the model instantiates without errors.
        encoder_layers=mcfg.decoder_layers,
        encoder_attention_heads=mcfg.decoder_attention_heads,
        encoder_ffn_dim=mcfg.decoder_ffn_dim,
        max_position_embeddings=mcfg.decoder_max_position_embeddings,
        dropout=mcfg.decoder_dropout,
        attention_dropout=mcfg.decoder_dropout,
        activation_dropout=mcfg.decoder_dropout,
        is_decoder=True,
        add_cross_attention=True,
        is_encoder_decoder=False,
        scale_embedding=True,
        pad_token_id=pad_id,
        bos_token_id=bos_id,
        eos_token_id=eos_id,
        decoder_start_token_id=bos_id,
    )
    return _CompatBartForCausalLM(dec_cfg)
