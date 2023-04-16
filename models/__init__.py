# ---------------------------------------------------------
# Yuxuan Zhang
# Dept. of Electrical and Computer Engineering
# University of Florida
# ---------------------------------------------------------
# TODO: Add description
# ---------------------------------------------------------
from .Encoder import Encoder
from .Decoder import Decoder
from .U_Net import Model as U_Net
from lib.Module import Module


MODELS: dict[Module] = {
    "Encoder": Encoder,
    "Decoder": Decoder,
    "U_Net": U_Net,
}
