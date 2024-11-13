from logging import getLogger

from ..utils.import_utils import compare_transformers_version
from ._base import BaseGPTQForCausalLM
from ._base_mlm import BaseGPTQForCausalMLM

if compare_transformers_version("v4.28.0", op="ge"):
    from ..nn_modules.fused_llama_attn import FusedLlamaAttentionForQuantizedModel
    from ..nn_modules.fused_llama_mlp import FusedLlamaMLPForQuantizedModel
else:
    FusedLlamaAttentionForQuantizedModel = None
    FusedLlamaMLPForQuantizedModel = None

logger = getLogger(__name__)


class LlamaGPTQForCausalLM(BaseGPTQForCausalLM):
    layer_type = "LlamaDecoderLayer"
    layers_block_name = "model.layers"
    outside_layer_modules = ["model.embed_tokens", "model.norm"]
    inside_layer_modules = [
        ["self_attn.k_proj", "self_attn.v_proj", "self_attn.q_proj"],
        ["self_attn.o_proj"],
        ["mlp.up_proj", "mlp.gate_proj"],
        ["mlp.down_proj"],
    ]

    fused_attn_module_type = FusedLlamaAttentionForQuantizedModel
    fused_mlp_module_type = FusedLlamaMLPForQuantizedModel

# class MiniCPMVGPTQ_Llama3(BaseGPTQForCausalMLM):
#     layer_type = "LlamaDecoderLayer"
#     layers_block_name = "llm.model.layers"
#     outside_layer_modules = ["llm.model.embed_tokens", "llm.model.norm"]
#     inside_layer_modules = [
#         ["self_attn.k_proj", "self_attn.v_proj", "self_attn.q_proj"],
#         ["self_attn.o_proj"],
#         ["mlp.up_proj", "mlp.gate_proj"],
#         ["mlp.down_proj"],
#     ]

#     fused_attn_module_type = FusedLlamaAttentionForQuantizedModel
#     fused_mlp_module_type = FusedLlamaMLPForQuantizedModel

# __all__ = ["LlamaGPTQForCausalLM","MiniCPMVGPTQ_Llama3"]

class MiniCPMVGPTQ_Llama3(BaseGPTQForCausalMLM):
    # 仅仅llm
    # layer_type = "LlamaDecoderLayer"
    # layers_block_name = "llm.model.layers"
    # outside_layer_modules = ["llm.model.embed_tokens", "llm.model.norm"]
    # inside_layer_modules = [
    #     ["self_attn.k_proj", "self_attn.v_proj", "self_attn.q_proj"], # 这里是不是会把vit也量了
    #     ["self_attn.o_proj"],
    #     ["mlp.up_proj", "mlp.gate_proj"],
    #     ["mlp.down_proj"],
    # ]
    # vlm
    layer_type = "LlamaDecoderLayer"
    layers_block_name = ""
    outside_layer_modules = ["llm.model.embed_tokens", "llm.model.norm", "vpm.embeddings", "vpm.post_layernorm"]
    inside_layer_modules = [
        ["self_attn.k_proj", "self_attn.v_proj", "self_attn.q_proj"], # 这里是不是会把vit也量了
        ["self_attn.o_proj"],
        ["mlp.up_proj", "mlp.gate_proj"],
        ["mlp.down_proj"],
        # ["self_attn.k_proj", "self_attn.v_proj", "self_attn.q_proj"],
        ["self_attn.out_proj"],
        ["mlp.fc1", "mlp.fc2"],
    ]
    fused_attn_module_type = FusedLlamaAttentionForQuantizedModel
    fused_mlp_module_type = FusedLlamaMLPForQuantizedModel
