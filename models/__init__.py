# ---------------------------------------------------------
# Yuxuan Zhang
# Dept. of Electrical and Computer Engineering
# University of Florida
# ---------------------------------------------------------
# TODO: Add description
# ---------------------------------------------------------
from .Encoder import Encoder
from .Decoder import Decoder
from .AutoEncoder import VisualAE, SpikeAE
from lib.Module import Module


MODELS: dict[Module] = {
    # ===== Combination =====
    "VisualAE": VisualAE,
    "SpikeAE": SpikeAE,
    # ===== Standalone ======
    "Encoder": Encoder,
    "Decoder": Decoder,
}
