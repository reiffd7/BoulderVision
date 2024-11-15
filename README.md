# BoulderVision: Climbing Movement Analysis

## Overview
BoulderVision is a computer vision tool that analyzes climbing movements in videos. It tracks a climber's movements, detects holds, and provides real-time visualization of movement patterns and velocity metrics.


[![IMAGE ALT TEXT HERE](https://img.youtube.com/vi/ClsECogdT7A/0.jpg)](https://www.youtube.com/watch?v=ClsECogdT7A)


## Features
- **Pose Detection**: Tracks 17 key body points throughout the climbing sequence
- **Hold Detection**: Identifies climbing holds in the frame
- **Movement Analysis**: 
  - Tracks cumulative movement over time
  - Calculates velocity ratios for movement analysis
  - Visualizes movement patterns with temporal color coding
- **Real-time Visualization**:
  - Multi-view display showing tracking, keypoints, and heatmaps
  - Live plotting of movement metrics
  - Interactive display of hold-body point matching

## Requirements

Core dependencies:
- OpenCV
- NumPy
- Plotly
- Supervision
- Roboflow Inference SDK
- PIL
- matplotlib
- tqdm

## Setup
1. Clone the repository
2. Install dependencies
3. Set up your Roboflow credentials:
   ```python
   API_URL = "https://boulder-vision.roboflow.cloud"
   API_KEY = "your_api_key"
   ```

## Usage
```python
from detect_holds_workflow import ClimbingAnalyzer

### Initialize analyzer
analyzer = ClimbingAnalyzer(
api_url="your_api_url",
api_key="your_api_key",
workspace_name="your_workspace",
workflow_id="your_workflow_id"
)

### Process a video
video_path = "path/to/your/video.mp4"
analyzer.process_video(video_path)

```
## Movement Analysis Metrics

The system calculates three key metrics to analyze movement patterns:

![Movement Analysis Visualization](assets/keypoint_movement_visualization.jpg)


#### 1. Trajectory Similarity (trajectory_cos_sim)
- Measures how closely the current movement follows the expected path using cosine similarity
- Range: [-1 to 1]
  - 1: Movements are identical in direction
  - 0: Movements are perpendicular
  - -1: Movements are in opposite directions
- Useful for detecting if a climber is following a similar path to previous attempts

#### 2. Velocity Ratio (velocity_ratio)
- Compares the current movement speed to the average speed over the previous window
- Interpretation:
  - > 1: Moving faster than the average historical speed
  - = 1: Moving at the same speed as the average
  - < 1: Moving slower than the average historical speed
- Helps identify acceleration/deceleration patterns and potential resting points

#### 3. Cumulative Distance (cumulative_distance)
- Total distance traveled by all keypoints over the analysis window
- Measured in pixels (or the units of your coordinate system)
- Higher values indicate more overall movement
- Useful for:
  - Detecting static vs. dynamic sequences
  - Identifying rest periods (low values)
  - Quantifying the amount of movement in a sequence

## Output
The script generates:
1. A processed video showing:
   - Climber tracking
   - Hold detection
   - Movement heatmaps
   - Real-time movement metrics
2. CSV file with movement data
3. Visualization overlays including:
   - Body keypoint tracking
   - Hold detection boxes
   - Movement metrics plots

## Configuration
Key parameters can be adjusted in the script:
- `start_seconds`: Start time for video processing
- `end_seconds`: End time for video processing
- `stride`: Frame processing stride
- Plot dimensions and styling
- Visualization overlay settings

## Contributing
Feel free to submit issues and enhancement requests!
