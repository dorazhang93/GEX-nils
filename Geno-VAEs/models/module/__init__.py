from .encoder import *
from .decoder import *

Encoders = {"EncoderMLP":MLPEncoder,
            'EncoderTransformer':Enformer,
            }

Decoders = {"DecoderMLP":MLPDecoder,
            "MLP":MLP,
            }