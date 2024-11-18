from logging import getLogger

from ..utils.import_utils import compare_transformers_version
from ._base_vit import BaseGPTQForVIT

if compare_transformers_version("v4.28.0", op="ge"):
    from ..nn_modules.fused_llama_attn import FusedLlamaAttentionForQuantizedModel
    from ..nn_modules.fused_llama_mlp import FusedLlamaMLPForQuantizedModel
else:
    FusedLlamaAttentionForQuantizedModel = None
    FusedLlamaMLPForQuantizedModel = None

logger = getLogger(__name__)


class VITGPTQ(BaseGPTQForVIT):
    layer_type = "SiglipEncoderLayer"
    layers_block_name = "vpm.encoder.layers"
    outside_layer_modules = ["vpm.embeddings", "vpm.post_layernorm"]
    inside_layer_modules = [
        ["self_attn.k_proj", "self_attn.v_proj", "self_attn.q_proj"],
        ["self_attn.out_proj"],
        ["mlp.fc1"],
        ["mlp.fc2"],
    ]
    #添加resampler模块量化，现在是吧resampler也量化了
    resampler_block_name = "resampler"
    resampler_layer_modules = ["kv_proj", "attn.out_proj"]

    fused_attn_module_type = FusedLlamaAttentionForQuantizedModel
    fused_mlp_module_type = FusedLlamaMLPForQuantizedModel
