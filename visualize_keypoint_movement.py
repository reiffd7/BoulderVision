import cv2
import numpy as np
from inference_sdk import InferenceHTTPClient
import supervision as sv
from tqdm import tqdm
import matplotlib.pyplot as plt
from utils import setup_video_processing
import os

edge_annotator = sv.EdgeAnnotator(color=sv.Color.GREEN, thickness=5)


def create_keypoint_movement_visualization(video_path, num_frames=15, start_seconds=19):
    # Setup video processing - capture video_info first
    frames_generator, total_frames, video_info = setup_video_processing(
        vid_path=video_path,
        start_seconds=start_seconds,
        end_seconds=start_seconds + 1,  # Temporary end_seconds
        stride=1
    )

    # Now we can use video_info.fps to calculate the correct end_seconds
    # Reset video processing with proper end_seconds
    frames_generator, total_frames, video_info = setup_video_processing(
        vid_path=video_path,
        start_seconds=start_seconds,
        end_seconds=start_seconds + (num_frames / video_info.fps),
        stride=1
    )

    # Initialize Roboflow client
    CLIENT = InferenceHTTPClient(
        api_url="http://localhost:5000",
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

    # Initialize variables for final metrics
    final_trajectory = "N/A"
    final_velocity = "N/A"
    final_distance = "N/A"

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
        keypoints_similarities = result[0]['keypoint_similarities']
        keypoints_obj = sv.KeyPoints.from_inference(keypoints)
        print(keypoints_similarities)
        # Draw keypoints and connections with color based on frame number
        color = tuple(map(int, colors[frame_idx]))  # Convert to BGR
        
        # Draw connections
        composite_frame = edge_annotator.annotate(
            scene=composite_frame,
            key_points=keypoints_obj
        )

        # Update history
        keypoints_history = result[0]['keypoints_history']

        # Update final metrics
        if keypoints_similarities['trajectory_similarity'] is not None:
            final_trajectory = f"{keypoints_similarities['trajectory_similarity']:.3f}"
        if keypoints_similarities['velocity_ratio'] is not None:
            final_velocity = f"{keypoints_similarities['velocity_ratio']:.3f}"
        if keypoints_similarities['cumulative_distance'] is not None:
            final_distance = f"{keypoints_similarities['cumulative_distance']:.3f}"

    # Add metrics text to the visualization
    metrics_height = 150
    metrics_bg = np.zeros((metrics_height, composite_frame.shape[1], 3), dtype=np.uint8)
    
    metrics_text = [
        f"Trajectory Similarity: {final_trajectory}",
        f"Velocity Ratio: {final_velocity}",
        f"Cumulative Distance: {final_distance}"
    ]
    
    for i, text in enumerate(metrics_text):
        cv2.putText(metrics_bg, text, (10, 30 + (i * 30)), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    # Combine visualization, legend, and metrics
    final_visualization = np.vstack([composite_frame, metrics_bg])

    # Save visualization
    cv2.imwrite('keypoint_movement_visualization.jpg', final_visualization)
    print("Visualization saved as 'keypoint_movement_visualization.jpg'")

if __name__ == "__main__":
    video_path = '/users/danielreiff/Downloads/IMG_3841.MOV'  # Update this path
    create_keypoint_movement_visualization(video_path)