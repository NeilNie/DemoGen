import os

os.environ["HF_HOME"] = "/svl/u/neilnie/cache"

from PIL import Image
from lang_sam import LangSAM
import numpy as np
from tqdm import tqdm
from termcolor import cprint


# Function to apply transparent masks on the image
def save_mask_results(image, masks, output_path):
    # Ensure the image is in RGBA mode (with transparency support)
    image = image_pil.convert("RGBA")

    # Iterate over the binary masks
    for mask in masks:
        # Convert mask to the same size as the image
        mask_resized = Image.fromarray(mask).resize(image.size, Image.NEAREST).convert("L")

        # Create an RGBA mask with the same size
        mask_rgba = mask_resized.convert("RGBA")

        # Get the alpha channel from the mask to apply transparency
        mask_rgba = np.array(mask_rgba)
        alpha_channel = mask_rgba[:, :, 0]  # Assuming the mask is single-channel

        # Create a new transparent image with the same size as the input image
        transparent_overlay = np.zeros((image.height, image.width, 4), dtype=np.uint8)

        # Set the red, green, and blue channels to some color (e.g., red)
        transparent_overlay[:, :, 0] = 255  # Red channel
        transparent_overlay[:, :, 1] = 0  # Green channel
        transparent_overlay[:, :, 2] = 0  # Blue channel

        # Set the alpha channel to apply transparency from the mask
        transparent_overlay[:, :, 3] = alpha_channel

        # Convert the overlay to an Image object
        overlay_image = Image.fromarray(transparent_overlay, "RGBA")

        # Composite the overlay with the original image
        image = Image.alpha_composite(image, overlay_image)

    # Save the resulting image
    image.save(output_path, "PNG")


# Function to save a binary mask as an image
def save_binary_mask(mask, output_path):
    # Convert the binary mask to a PIL Image object
    # Convert the mask to uint8 type (0 or 255)
    mask_image = Image.fromarray((mask * 255).astype(np.uint8))  # Multiply by 255 to make 1 white and 0 black

    # Save the mask as an image (you can choose PNG, JPEG, etc.)
    mask_image.save(output_path)


# ------------------------------------------------------------------------------
model = LangSAM(sam_type="sam2.1_hiera_large")
text_prompts = ["green book on the table", "brown wooden bookshelf"]
img_dir = "/svl/u/neilnie/workspace/vlm-policy-learning/baselines/DemoGen/data/sam_mask/bookshelf"
# ------------------------------------------------------------------------------


for demo in tqdm(os.listdir(img_dir)):
    demo_dir = os.path.join(img_dir, demo)
    image_pil = Image.open(os.path.join(demo_dir, "source.png")).convert("RGB")

    for prompt in text_prompts:
        results = model.predict([image_pil], [prompt])

        masks = results[0]["masks"]
        scores = results[0]["scores"]
        max_idx = np.argmax(scores)
        mask = masks[max_idx]
        label = results[0]["labels"][max_idx]
        score = scores[max_idx]

        cprint(f"prompt: {prompt}, label: {label}, score: {score}", "blue")

        save_binary_mask(mask, f"{demo_dir}/{prompt}.jpg")

cprint("Done", "green")
