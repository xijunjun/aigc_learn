# export HF_ENDPOINT="https://hf-mirror.com"
from diffusers import DiffusionPipeline
import torch

# load both base & refiner
base = DiffusionPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16, variant="fp16", use_safetensors=True,cache_dir='./models'
)
base.to("cuda")
refiner = DiffusionPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-refiner-1.0",
    text_encoder_2=base.text_encoder_2,
    vae=base.vae,
    torch_dtype=torch.float16,
    use_safetensors=True,
    variant="fp16",cache_dir='./models'
)
refiner.to("cuda")

# Define how many steps and what % of steps to be run on each experts (80/20) here
n_steps = 40
high_noise_frac = 0.8

prompt = [ 
    # "A majestic lion jumping from a big stone at night"
    # 'a dog drinking a pint of beer,cannon,real'
    # 'Two cat teams are playing a basketball game, one team is all black,another team is all white'
    # 'Two basketball teams are playing a game, with the players having cat heads and human bodies.'
    'Portrait, half body, Girl, black short hair, ((((Ultra-HD-details, 60+fps-details)))),dslr,cannon'
][0]


# prompt="Astronaut in a jungle, cold color palette, muted colors, detailed, 8k",
negative_prompt=[
    # "ugly, deformed, disfigured, poor details, bad anatomy,3d, cartoon, anime, sketches, (worst quality:2), (low quality:2)"
    # 'cartoon'
    '(((exotics, exot, 10-exotic-fingers, exotic-hands, exotic-limbs, exotic-joints, exotic-toes, exotic-appearances, 4-finger-hands, hiding-hands, hiding-fingers))) far-away-worsening, far-away-ugliness, CGI, kiddy-grade, 3D-rendering,, (((exotics, exot, 10-exotic-fingers, exotic-hands, exotic-limbs, exotic-joints, exotic-toes, exotic-appearances, 4-finger-hands, hiding-hands, hiding-fingers))) far-away-worsening, far-away-ugliness, CGI, kiddy-grade, 3D-rendering,'
][0]


guidance_scale=7.5

# run both experts
image = base(
    prompt=prompt,
    negative_prompt=negative_prompt,
    guidance_scale=guidance_scale,
    num_inference_steps=n_steps,
    denoising_end=high_noise_frac,
    output_type="latent",
).images
image = refiner(
    prompt=prompt,
    negative_prompt=negative_prompt,
    guidance_scale=guidance_scale,
    num_inference_steps=n_steps,
    denoising_start=high_noise_frac,
    image=image,
).images[0]
image.save('test-paper.jpg')