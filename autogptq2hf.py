import os
from safetensors.torch import load
import shutil
import torch

# for bits in [2,4,8], per-channel quantization

def get_fake_weight(bits, qzeros, scales, qweight, group_size):
    wf = torch.tensor(list(range(0, 32, bits)), dtype=torch.int32).unsqueeze(0)
    zeros = torch.bitwise_right_shift(
        torch.unsqueeze(qzeros, 2).expand(-1, -1, 32 // bits),
        wf.unsqueeze(0),
    ).to(torch.int16 if bits == 8 else torch.int8)

    zeros = zeros + 1
    zeros = torch.bitwise_and(
        zeros, (2**bits) - 1
    )  # NOTE: It appears that casting here after the `zeros = zeros + 1` is important.

    zeros = zeros.reshape(-1, 1, zeros.shape[1] * zeros.shape[2])

    scales = scales
    scales = scales.reshape(-1, 1, scales.shape[-1])

    weight = torch.bitwise_right_shift(
        torch.unsqueeze(qweight, 1).expand(-1, 32 // bits, -1),
        wf.unsqueeze(-1),
    ).to(torch.int16 if bits == 8 else torch.int8)
    weight = torch.bitwise_and(weight, (2**bits) - 1)
    weight = weight.reshape(-1, weight.shape[2])
    weight = scales * (weight - zeros)
    weight = weight.squeeze(0).T
    return weight

# real_path = '/data1/liyx/Projects/AutoGPTQ_mlm/Models/MiniCPM-3o-1B-sft-v1-vit-w8-pc-c256-llm-w4-pc/model.safetensors'
real_path = '/data1/liyx/Projects/AutoGPTQ_mlm/Models/MiniCPM-3o-1B-sft-v1-vitw8/gptq_model-8bit--1g.safetensors'
fake_path = '/data1/liyx/Projects/AutoGPTQ_mlm/Models/MiniCPM-3o-1B-sft-v1-vitw8-fakequant'
source_dir = '/data1/liyx/Models/MiniCPM-3o-1B-sft-v1-fixshape'

os.makedirs(fake_path, exist_ok=True)
with open(real_path, "rb") as f:
    file = f.read()
loaded_data = load(file)
result_dict = {}
for key, value in loaded_data.items():
    # print("key: ", key)
    # print(value.shape)
    common_part = '.'.join(key.split('.')[:-1])  
    last_part = key.split('.')[-1]
    bits = 4 if "llm" in key else 8
    if 'qweight' in last_part:
        print(f"fake quanting {key}")
        qweight = loaded_data.get(f"{common_part}.qweight")
        qzeros = loaded_data.get(f"{common_part}.qzeros")
        scales = loaded_data.get(f"{common_part}.scales")
        weight = get_fake_weight(bits, qzeros, scales, qweight, -1)
        result_dict[common_part+".weight"] = weight
    elif "qzeros" in last_part or "scales" in last_part or "g_idx" in last_part:
        continue
    else:
        result_dict[common_part+f".{last_part}"] =value

torch.save(result_dict, os.path.join(fake_path, "pytorch_model.bin"))

os.makedirs(fake_path, exist_ok=True)
for root, dirs, files in os.walk(source_dir):
    for file in files:
        if file == 'pytorch_model.bin':
            continue 
        source_file = os.path.join(root, file)
        relative_path = os.path.relpath(source_file, source_dir)
        destination_file = os.path.join(fake_path, relative_path)
        os.makedirs(os.path.dirname(destination_file), exist_ok=True)
        shutil.copy2(source_file, destination_file)
