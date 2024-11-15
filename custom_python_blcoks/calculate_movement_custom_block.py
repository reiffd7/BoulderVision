import numpy as np

def compute_trajectory_similarities(history, new_points, window_size):
    """Compare movement trajectories over multiple frames"""
    print("History shape:", history.shape)
    print("New points shape:", new_points.shape)
    print("Window size:", window_size)
    
    # Default return values
    default_similarities = {
        'trajectory_similarity': None,
        'velocity_ratio': None,
        'cumulative_distance': None
    }
    
    # Check if we have enough non-zero keypoints in history
    non_zero_frames = np.any(history != 0, axis=(2, 3))  # Check across keypoint coordinates
    valid_frame_count = np.sum(non_zero_frames)
    print("Valid frame count:", valid_frame_count)
    
    if valid_frame_count < window_size or len(new_points) == 0:
        print("Returning default due to insufficient frames or no new points")
        return default_similarities
        
    # Calculate trajectory vectors (movement over multiple frames)
    history_trajectory = history[-1] - history[-window_size]  # Total movement over window
    new_trajectory = new_points - history[-window_size]       # Movement from start of window to new points
    
    print("History trajectory shape:", history_trajectory.shape)
    print("New trajectory shape:", new_trajectory.shape)
    
    # For velocity changes (acceleration), we can look at sequential differences
    history_velocities = np.diff(history[-window_size:], axis=0)  # Shape: (window_size-1, 1, 17, 2)
    new_velocity = new_points - history[-1]
    
    # Flatten for similarity calculations
    history_trajectory_flat = history_trajectory.reshape(-1)
    new_trajectory_flat = new_trajectory.reshape(-1)
    
    # 1. Overall trajectory similarity (cosine)
    trajectory_cos_sim = np.dot(history_trajectory_flat, new_trajectory_flat) / (
        np.linalg.norm(history_trajectory_flat) * np.linalg.norm(new_trajectory_flat) + 1e-8
    )
    
    # 2. Average velocity magnitude
    avg_history_velocity = np.mean(np.linalg.norm(history_velocities.reshape(window_size-1, -1), axis=1))
    new_velocity_magnitude = np.linalg.norm(new_velocity.reshape(-1))
    velocity_ratio = new_velocity_magnitude / (avg_history_velocity + 1e-8)
    
    # 3. Cumulative distance traveled
    history_cumulative_distance = np.sum(np.linalg.norm(history_velocities.reshape(window_size-1, -1), axis=1))
    
    return {
        'trajectory_similarity': trajectory_cos_sim,
        'velocity_ratio': velocity_ratio,
        'cumulative_distance': history_cumulative_distance
    }

def run(self, keypoints_history, cur_frame_keypoints, frame_num, window_size) -> BlockResult:
    window_size = int(window_size)
    keypoints_history = np.array(keypoints_history)

    # Get current keypoints
    keypoints_xy = cur_frame_keypoints.data['keypoints_xy']

    # If we have current keypoints, find the closest one to our history
    if len(keypoints_xy) > 0:
        last_known_position = np.array(keypoints_history[-1])
        distances = []
        for kp in keypoints_xy:
            kp = np.array(kp)
            # No need to reshape here since the shapes are already correct
            diff = kp - last_known_position
            distance = np.mean(np.sum(diff**2, axis=1))
            distances.append(distance)
        
        closest_idx = np.argmin(distances)
        print("closest_idx:", closest_idx)
        print("keypoints_xy shape before:", np.array(keypoints_xy).shape)
        keypoints_xy = np.array(keypoints_xy[closest_idx])
        print("keypoints_xy shape after array:", keypoints_xy.shape)
        keypoints_xy = keypoints_xy[np.newaxis, ...]
        print("keypoints_xy shape after newaxis:", keypoints_xy.shape)
        print("keypoints_xy:", keypoints_xy)
    
    # Always call compute_trajectory_similarities
    similarities = compute_trajectory_similarities(keypoints_history, keypoints_xy, window_size)
    
    # Update history with current keypoints
    if len(keypoints_xy) == 0:
        keypoints_xy = keypoints_history[-1]
    keypoints_history = np.roll(keypoints_history, -1, axis=0)
    keypoints_history[-1] = keypoints_xy
    return {'similarities': similarities, 'keypoints_history': keypoints_history}