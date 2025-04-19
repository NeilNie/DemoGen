import pickle
import os
import shutil
import cv2
from tqdm import tqdm
from termcolor import cprint


demo_dirs = [
    "/svl/u/neilnie/short_hang_mug_front",
    "/svl/u/neilnie/short_hang_mug_top_left",
    "/svl/u/neilnie/short_hang_mug_top_right",
    "/svl/u/neilnie/short_hang_mug_bottom_left",
    "/svl/u/neilnie/short_hang_mug_bottom_right",
]

demo_name = "mug_tree"
output_dir = "/svl/u/neilnie/workspace/vlm-policy-learning/baselines/DemoGen/data/sam_mask"

# create output directory
output_dir = os.path.join(output_dir, demo_name)
shutil.rmtree(output_dir, ignore_errors=True)
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

demo_counter = 0
files = []

for demo_dir in tqdm(demo_dirs, desc="Processing demos"):
    for file in sorted(os.listdir(demo_dir)):

        if not file.endswith(".pkl"):
            continue
        demo_file = os.path.join(demo_dir, file)
        with open(demo_file, "rb") as f:
            demo_data = pickle.load(f)
        
        this_demo_dir = os.path.join(output_dir, str(demo_counter))
        if not os.path.exists(this_demo_dir):
            os.makedirs(this_demo_dir)
        
        first_frame = demo_data["camera_stream_0"]["rgb_frames"][0]
        cv2.imwrite(os.path.join(this_demo_dir, "source.png"), first_frame)
        demo_counter += 1

        files.append(demo_file)
        # TODO: optional break here
        print(demo_data["temporal_segmentation"])
        break

cprint("Done", "green")
cprint(f"output dir: {output_dir}", "yellow")
print(files)