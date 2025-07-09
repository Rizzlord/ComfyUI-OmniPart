import torch
import torch.utils.checkpoint
from torch import nn

from transformers import AutoModelForCausalLM, AutoConfig
from transformers.models.opt.modeling_opt import OPTForCausalLM, OPTModel, OPTDecoder, OPTConfig

from transformers.utils import logging
from typing import Optional, Union

from transformers.generation.logits_process import LogitsProcessorList
from transformers.generation.utils import GenerateNonBeamOutput, GenerateEncoderDecoderOutput, GenerateDecoderOnlyOutput
from transformers.generation.stopping_criteria import StoppingCriteriaList
from transformers.generation.configuration_utils import GenerationConfig
from transformers.generation.streamers import BaseStreamer

logger = logging.get_logger(__name__)

class BBoxOPTConfig(OPTConfig):
    model_type = "mesh_opt"

class BBoxOPTDecoder(OPTDecoder):
    config_class = BBoxOPTConfig

class BBoxOPTModel(OPTModel):
    config_class = BBoxOPTConfig
    def __init__(self, config: BBoxOPTConfig):
        super(OPTModel, self).__init__(config)
        self.decoder = BBoxOPTDecoder(config)
        # Initialize weights and apply final processing
        self.post_init()

class BBoxOPT(OPTForCausalLM):
    config_class = BBoxOPTConfig

    def __init__(self, config: BBoxOPTConfig):
        super(OPTForCausalLM, self).__init__(config)
        self.model = BBoxOPTModel(config)

        # the lm_head weight is automatically tied to the embed tokens weight
        self.lm_head = nn.Linear(config.word_embed_proj_dim, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()
    

AutoConfig.register("mesh_opt", BBoxOPTConfig)
AutoModelForCausalLM.register(BBoxOPTConfig, BBoxOPT)
