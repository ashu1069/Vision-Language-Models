import torch
from torch import nn
from typing import Optional, Tuple, List
from torch.nn import CrossEntropyLoss
import math
from siglip import SiglipVisionConfig, SiglipVisionModel

class PaliGemmaForConditionalGeneration(nn.Module):
    def __init__(self, config: PaliGemmaConfig):
    super().__init__()
    self.config = config
    self.vision_tower = SiglipVisionModel(config.vision_config)
    self.multi_modal_projector = PaliGemmaMultiModalProjector(config)
    self.vocab_size = config.vocab_size

    language_model = GemmaForCausalLM(config.text_config)
    self.language_model = language_model
