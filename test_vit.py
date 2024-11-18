import datasets
import random
import numpy as np
import torch
from transformers import AutoTokenizer, TextGenerationPipeline
from auto_gptq import AutoGPTQForVIT, BaseQuantizeConfig
import logging
from transformers import AutoProcessor


def get_ScienceQA(nsamples, seed, seqlen, processor):
    import torch.nn.functional as F
    dataset = datasets.load_from_disk("/data1/liyx/Datasets/ScienceQA-2")['train']
    dataset = dataset.shuffle(seed=seed)
    rng = random.Random(42)
    samples, num_tokens = [], 0
    prompts_lists = []
    input_images_lists = []
    for index, _data in enumerate(dataset):
        promt = _data["question"]
        image_file = _data["image"]
        image = np.array(image_file)
        if image_file is None:
            nsamples = nsamples+1
            continue
        msgs = [{'role': 'user', 'content': "(<image>./</image>)\n"+ promt}]
        prompts_lists.append(processor.tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True))
        input_images_lists.append([image])
        if index >= nsamples-1:
            break
    # return prompts_lists,input_images_lists
     
    inputs = processor(
        prompts_lists,
        input_images_lists,
        max_slice_nums=processor.image_processor.max_slice_nums,
        # max_slice_nums=9,
        use_image_id=processor.image_processor.use_image_id,
        return_tensors="pt",
        max_length=8192
    )
    # return inputs
    traindataset = []
    for _ in range(inputs["input_ids"].size(0)):
        input_ids = inputs["input_ids"].select(0, _).unsqueeze(0) 
        attention_mask = inputs["attention_mask"].select(0, _).unsqueeze(0) 
        pixel_values = [inputs["pixel_values"][_]]
        image_sizes = [inputs["image_sizes"][_]]
        image_bound = [inputs["image_bound"][_]]
        tgt_sizes = [inputs["tgt_sizes"][_]]
        traindataset.append({"input_ids": input_ids, 
                             "attention_mask": attention_mask,
                             "pixel_values": pixel_values,
                             "image_sizes": image_sizes,
                             "image_bound": image_bound,
                             "tgt_sizes": tgt_sizes})

    return traindataset


logging.basicConfig(
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s", level=logging.INFO, datefmt="%Y-%m-%d %H:%M:%S"
)

pretrained_model_dir = "/data1/liyx/Models/MiniCPM-3o-1B-sft-v1-fixshape-dev"
quantized_model_dir = "Models/MiniCPM-3o-1B-sft-v1-vitw8"

tokenizer = AutoTokenizer.from_pretrained(pretrained_model_dir, use_fast=True, trust_remote_code=True)
quantize_config = BaseQuantizeConfig(
    bits=8,  # quantize model to 4-bit
    group_size=-1,  # it is recommended to set the value to 128
    desc_act=True,  # set to False can significantly speed up inference but the perplexity may slightly bad
)
# traindataset, testenc = get_wikitext2(128, 0, model.seqlen, tokenizer)
# load un-quantized model, by default, the model will always be loaded into CPU memory
model = AutoGPTQForVIT.from_pretrained(pretrained_model_dir, quantize_config)
model.model.processor = AutoProcessor.from_pretrained(pretrained_model_dir, trust_remote_code=True)

traindataset = get_ScienceQA(2048, 1234, model.seqlen, model.model.processor)
model.quantize(traindataset)
model.save_quantized(quantized_model_dir)

# model_quant = AutoGPTQForVIT.from_quantized(quantized_model_dir, device="cuda:0", use_triton=False)
