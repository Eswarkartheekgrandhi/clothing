from diffusers import StableDiffusionPipeline, ControlNetModel
import torch
from PIL import Image

def generate_image(prompt, canny_image):
    pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16).to("cuda")  # Or your model

    # Load ControlNet model (Canny) - Correct way
    controlnet = ControlNetModel.from_pretrained("lllyasviel/control_v11p_sd15_canny", torch_dtype=torch.float16).to("cuda")

    control_pipe = StableDiffusionPipeline(
        vae=pipe.vae, text_encoder=pipe.text_encoder, tokenizer=pipe.tokenizer, unet=pipe.unet, controlnet=controlnet, scheduler=pipe.scheduler,
        torch_dtype=torch.float16
    ).to("cuda")

    image = control_pipe(prompt, image=canny_image).images[0]  # Or add controlnet_conditioning_scale
    return image

if __name__ == "__main__":
    try:
        with open("Approach1\prompt.txt", "r") as f:
            prompt = f.read().strip()
    except FileNotFoundError:
        print("Error: prompt.txt not found. Run color_extraction.py first.")
        exit()

    canny_image = Image.open("Approach1\canny_edges.png").convert("RGB") # Ensure RGB

    generated_image = generate_image(prompt, canny_image)
    generated_image.save("generated_fabric.jpg")
    print("Image generated successfully!")