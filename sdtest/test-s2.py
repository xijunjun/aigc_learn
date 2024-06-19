from diffusers import StableDiffusionXLPipeline, StableDiffusionXLImg2ImgPipeline,DiffusionPipeline
import torch

# pipeline = StableDiffusionXLPipeline.from_pretrained(
#     "stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16, variant="fp16", use_safetensors=True,cache_dir='./models'
# ).to("cuda")
base = DiffusionPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16, variant="fp16", use_safetensors=True,cache_dir='./models'
).to("cuda")


refiner = StableDiffusionXLImg2ImgPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-refiner-1.0", torch_dtype=torch.float16, use_safetensors=True, variant="fp16",cache_dir='./models'
).to("cuda")

# prompt = "A majestic lion jumping from a big stone at night"

prompt = [
    # "Astronaut in a jungle, cold color palette, muted colors, detailed, 8k"
'A majestic elf queen standing in an enchanted forest, with flowing silver hair, intricate golden armor, and a glowing magical staff.'
    # 'A futuristic cityscape at night with towering skyscrapers, neon lights, flying cars, and bustling streets filled with diverse alien species.'
][0]

image = base(
    prompt=prompt,
    num_inference_steps=40,
    denoising_end=0.8,
    output_type="latent",
).images
image = refiner(
    prompt=prompt,
    num_inference_steps=40,
    denoising_start=0.8,
    image=image,
).images[0]
image.save('s2.jpg')