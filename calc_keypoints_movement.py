from inference import InferencePipeline
import cv2
from utils import setup_video_processing
import matplotlib.pyplot as plt
from inference_sdk import InferenceHTTPClient
import supervision as sv
from tqdm import tqdm
import numpy as np






frames_generator, total_frames, video_info = setup_video_processing(
    vid_path='/users/danielreiff/Downloads/IMG_3841.MOV',
    start_seconds=31,
    end_seconds=32,
    stride=1
)


CLIENT = InferenceHTTPClient(
    api_url="http://localhost:5000", # use local inference server
    api_key="CStUD4FGBWGrkAzyK7ou"
)

history_size = 5

# Initialize as a nested list with dimensions [history_size][1][17][2]
keypoints_history = [[[[0.0, 0.0] for _ in range(17)] for _ in range(1)] for _ in range(history_size)]

for frame in tqdm(frames_generator):
    result = CLIENT.run_workflow(
        workspace_name="daniels-workspace-dnmsg",
        workflow_id="detect-holds-v8",
        images={
            "image": frame,
        },
        parameters={
            "keypoints_history": keypoints_history
        }
    )

    print(np.array(keypoints_history).shape)
    keypoints_history = result[0]['prev_frames_keypoints']

