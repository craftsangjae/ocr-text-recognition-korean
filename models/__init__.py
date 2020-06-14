from tensorflow.keras.utils import get_custom_objects
from models.layers import *
from models.optimizer import *
from models.losses import *

# Custom Layer 구성하기
get_custom_objects().update({
    "ConvFeatureExtractor" : ConvFeatureExtractor,
    "ResidualConvFeatureExtractor": ResidualConvFeatureExtractor,
    "Map2Sequence" : Map2Sequence,
    "BGRUEncoder" : BGRUEncoder,
    "CTCDecoder" : CTCDecoder,
    "DotAttention": DotAttention,
    "JamoCompose": JamoCompose,
    'JamoDeCompose': JamoDeCompose,
    "JamoEmbedding": JamoEmbedding,
    "JamoClassifier": JamoClassifier,
    "TeacherForcing": TeacherForcing
})

# Custom Optimizer 구성하기
get_custom_objects().update({'AdamW': AdamW,
                             'RectifiedAdam': RectifiedAdam})

# Custom Loss 구성하기
get_custom_objects().update({
    'ctc_loss': ctc_loss,
    "JamoCategoricalCrossEntropy": JamoCategoricalCrossEntropy})
