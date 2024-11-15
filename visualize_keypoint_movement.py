import cv2
import numpy as np
from inference_sdk import InferenceHTTPClient
import supervision as sv
from tqdm import tqdm
import matplotlib.pyplot as plt
from utils import setup_video_processing
import os

def create_keypoint_movement_visualization(video_path, num_frames=10, start_seconds=19):
    # Setup video processing
    frames_generator, total_frames, video_info = setup_video_processing(
        vid_path=video_path,
        start_seconds=start_seconds,
        end_seconds=start_seconds + (num_frames / video_info.fps),
        stride=1
    )

    # Initialize Roboflow client
    CLIENT = InferenceHTTPClient(
        api_url="https://boulder-vision.roboflow.cloud",
        api_key=os.getenv("ROBOFLOW_API_KEY")
    )

    # Initialize keypoints history
    history_size = 10
    keypoints_history = [[[[0.0, 0.0] for _ in range(17)] for _ in range(1)] for _ in range(history_size)]
    
    # Create empty frame to store visualization
    first_frame = next(frames_generator)
    composite_frame = first_frame.copy()
    
    # Setup colors for temporal visualization (from red to blue)
    colors = plt.cm.RdYlBu(np.linspace(0, 1, num_frames))
    colors = (colors[:, :3] * 255).astype(np.uint8)

    # Process frames
    frames_generator, total_frames, _ = setup_video_processing(
        vid_path=video_path,
        start_seconds=start_seconds,
        end_seconds=start_seconds + (num_frames / video_info.fps),
        stride=1
    )

    print("Processing frames...")
    for frame_idx, frame in tqdm(enumerate(frames_generator)):
        if frame_idx >= num_frames:
            break

        # Get keypoint detections
        result = CLIENT.run_workflow(
            workspace_name="daniels-workspace-dnmsg",
            workflow_id="detect-holds-v17",
            images={"image": frame},
            parameters={
                "keypoints_history": keypoints_history,
                "frame_num": frame_idx
            },
            use_cache=False
        )

        keypoints = result[0]['keypoint_outputs']
        keypoints_obj = sv.KeyPoints.from_inference(keypoints)
        
        # Draw keypoints and connections with color based on frame number
        color = tuple(map(int, colors[frame_idx]))  # Convert to BGR
        
        # Draw connections
        for connection in sv.KeyPoints.EDGES:
            pt1 = tuple(map(int, keypoints_obj.data[0][connection[0]]))
            pt2 = tuple(map(int, keypoints_obj.data[0][connection[1]]))
            cv2.line(composite_frame, pt1, pt2, color, 2)
        
        # Draw keypoints
        for point in keypoints_obj.data[0]:
            cv2.circle(composite_frame, tuple(map(int, point)), 5, color, -1)

        # Update history
        keypoints_history = result[0]['keypoints_history']

    # Add color legend
    legend_height = 50
    legend_width = composite_frame.shape[1]
    legend = np.zeros((legend_height, legend_width, 3), dtype=np.uint8)
    
    for i in range(legend_width):
        color_idx = int((i / legend_width) * (num_frames - 1))
        legend[:, i] = colors[color_idx]

    # Add text labels
    cv2.putText(legend, 'Earlier', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(legend, 'Later', (legend_width - 100, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    # Combine visualization and legend
    final_visualization = np.vstack([composite_frame, legend])

    # Save visualization
    cv2.imwrite('keypoint_movement_visualization.jpg', final_visualization)
    print("Visualization saved as 'keypoint_movement_visualization.jpg'")

if __name__ == "__main__":
    video_path = '/users/danielreiff/Downloads/IMG_3841.MOV'  # Update this path
    create_keypoint_movement_visualization(video_path)