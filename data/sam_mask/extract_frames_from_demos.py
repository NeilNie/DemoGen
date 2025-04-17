import pickle
import os
import shutil
import cv2
from tqdm import tqdm
from termcolor import cprint


demo_dirs = [
    "/svl/u/neilnie/bimanual_new_bookshelf_left_processed_new_new/",
    "/svl/u/neilnie/bimanual_new_bookshelf_right_processed_2"
]

demo_name = "bookshelf"
output_dir = "/svl/u/neilnie/workspace/vlm-policy-learning/baselines/DemoGen/data/sam_mask"


# create output directory
output_dir = os.path.join(output_dir, demo_name)
shutil.rmtree(output_dir, ignore_errors=True)
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

demo_counter = 0

for demo_dir in tqdm(demo_dirs, desc="Processing demos"):
    for file in os.listdir(demo_dir):

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


cprint("Done", "green")
cprint("output dir: ", output_dir, "yellow")
