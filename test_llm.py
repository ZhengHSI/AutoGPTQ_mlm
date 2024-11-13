import random
import numpy as np
import torch
from transformers import AutoTokenizer, TextGenerationPipeline
from auto_gptq import AutoGPTQForVIT, BaseQuantizeConfig, AutoGPTQForCausalMLM
import logging 
from transformers import AutoProcessor
from datasets import load_dataset, load_from_disk

def get_wikitext2(nsamples, seed, seqlen, tokenizer):
    # set seed
    random.seed(seed)
    np.random.seed(seed)
    torch.random.manual_seed(seed)

    # load dataset and preprocess
    # traindata = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    # testdata = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    traindata = load_dataset("/home/workspace/code/git/FlatQuant_mlm/datasets/wikitext", split="train")
    testdata = load_dataset("/home/workspace/code/git/FlatQuant_mlm/datasets/wikitext", split="test")
    trainenc = tokenizer("\n\n".join(traindata["text"]), return_tensors="pt")
    testenc = tokenizer("\n\n".join(testdata["text"]), return_tensors="pt")

    traindataset = []
    for _ in range(nsamples):
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        attention_mask = torch.ones_like(inp)
        traindataset.append({"input_ids": inp, "attention_mask": attention_mask})
    return traindataset, testenc

logging.basicConfig(
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s", level=logging.INFO, datefmt="%Y-%m-%d %H:%M:%S"
)

pretrained_model_dir = "/home/workspace/model/MiniCPM-3o-1B-sft-v1"
quantized_model_dir = "/home/workspace/model/MiniCPM-3o-1B-sft-v1-llm_w4_pc_c256"

tokenizer = AutoTokenizer.from_pretrained(pretrained_model_dir, use_fast=True, trust_remote_code=True)
quantize_config = BaseQuantizeConfig(
    bits=4,  # quantize model to 8-bit
    group_size=-1,  # it is recommended to set the value to -1
    desc_act=True,  # set to False can significantly speed up inference but the perplexity may slightly bad
)
# load un-quantized model, by default, the model will always be loaded into CPU memory
model = AutoGPTQForCausalMLM.from_pretrained(pretrained_model_dir, quantize_config)
model.model.processor = AutoProcessor.from_pretrained(pretrained_model_dir, trust_remote_code=True)
traindataset, testenc = get_wikitext2(256, 0, model.seqlen, tokenizer)
model.quantize(traindataset)
model.save_quantized(quantized_model_dir)

# model_quant = AutoGPTQForVIT.from_quantized(quantized_model_dir, device="cuda:0", use_triton=False)