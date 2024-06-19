
from diffusers import AutoPipelineForText2Image
import torch

pipeline_text2image = AutoPipelineForText2Image.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16, variant="fp16", use_safetensors=True,cache_dir='./models'
).to("cuda")

prompt = [
    # "Astronaut in a jungle, cold color palette, muted colors, detailed, 8k"
# 'A majestic elf queen standing in an enchanted forest, with flowing silver hair, intricate golden armor, and a glowing magical staff.'
    'A futuristic cityscape at night with towering skyscrapers, neon lights, flying cars, and bustling streets filled with diverse alien species.'
][0]
image = pipeline_text2image(prompt=prompt).images[0]
image.save('res2.jpg')
