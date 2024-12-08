import cv2
import numpy as np
import pandas as pd
import supervision as sv
import matplotlib.pyplot as plt
from tqdm import tqdm
from inference_sdk import InferenceHTTPClient
from utils import setup_video_processing
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import io
from PIL import Image
import os
import yaml

class ClimbingAnalyzer:
    def __init__(self, api_url, api_key, workspace_name, workflow_id):
        self.client = InferenceHTTPClient(api_url=api_url, api_key=api_key)
        self.workspace_name = workspace_name
        self.workflow_id = workflow_id
        
        # Load config for history size
        with open('config.yaml', 'r') as file:
            config = yaml.safe_load(file)
        self.history_size = config['analyzer']['history_size']
        
        self.keypoints_history = [[[[0.0, 0.0] for _ in range(17)] 
                                 for _ in range(1)] 
                                 for _ in range(self.history_size)]
        
        # Initialize annotators
        self.setup_annotators()
        
        # Initialize plotting
        self.setup_plotting()
        
    def setup_annotators(self):
        self.tracker = sv.ByteTrack()
        self.smoother = sv.DetectionsSmoother()
        self.box_annotator = sv.BoxAnnotator()
        self.label_annotator = sv.LabelAnnotator()
        self.heatmap_annotator = sv.HeatMapAnnotator(position=sv.geometry.core.Position.CENTER, opacity=0.6)
        self.trace_annotator = sv.TraceAnnotator(trace_length=10000000)
        self.edge_annotator = sv.EdgeAnnotator(color=sv.Color.GREEN, thickness=5)

    def setup_plotting(self):
        # Initialize data lists
        self.frame_numbers = []
        self.cumulative_movements = []
        self.velocity_ratios = []
        
        # Create plotly figure
        self.fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=('Cumulative Movement', 'Velocity Ratio'),
            vertical_spacing=0.12
        )
        
        # Update layout for transparency and size
        self.fig.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(
                color='white',
                size=8
            ),
            showlegend=False,
            margin=dict(l=30, r=10, t=20, b=10),
            height=200,
            width=500,   # Even wider width
            title_font_size=8
        )
        
        # Update axes for both subplots
        self.fig.update_xaxes(
            showgrid=False,
            title_text='Frame',
            color='white',
            title_font=dict(size=8),
            tickfont=dict(size=6)
        )
        
        self.fig.update_yaxes(
            showgrid=False,
            color='white',
            title_font=dict(size=8),
            tickfont=dict(size=6)
        )

    def run_inference(self, frame, frame_num):
        return self.client.run_workflow(
            workspace_name=self.workspace_name,
            workflow_id=self.workflow_id,
            images={"image": frame},
            parameters={
                "keypoints_history": self.keypoints_history,
                "frame_num": frame_num,
                "window_size": self.history_size
            },
            use_cache=False
        )

    def update_plots(self, frame_num, keypoint_similarities):
        if 'cumulative_distance' not in keypoint_similarities:
            return None
            
        self.frame_numbers.append(frame_num)
        self.cumulative_movements.append(keypoint_similarities['cumulative_distance'])
        self.velocity_ratios.append(keypoint_similarities.get('velocity_ratio', 0))
        
        # Update traces
        self.fig.data = []  # Clear previous traces
        
        # Add new traces
        self.fig.add_trace(
            go.Scatter(
                x=self.frame_numbers,
                y=self.cumulative_movements,
                line=dict(color='blue', width=2),
                opacity=0.8
            ),
            row=1, col=1
        )
        
        self.fig.add_trace(
            go.Scatter(
                x=self.frame_numbers,
                y=self.velocity_ratios,
                line=dict(color='red', width=2),
                opacity=0.8
            ),
            row=2, col=1
        )
        
        # Update axes titles
        self.fig.update_yaxes(title_text='Cumulative Movement', row=1, col=1)
        self.fig.update_yaxes(title_text='Velocity Ratio', row=2, col=1)

        
        # Convert to image
        img_bytes = self.fig.to_image(
            format="png",
            width=400,    # Match the wider layout width
            height=200,   # Match the layout height
            scale=2,
        )
        
        # Convert bytes to numpy array
        img = Image.open(io.BytesIO(img_bytes))
        plot_image = np.array(img)
        
        return cv2.cvtColor(plot_image, cv2.COLOR_RGBA2BGR)

    def draw_matches_text(self, frame, points_holds_matches):
        y_offset = 50
        x_offset = frame.shape[1] - 400
        line_height = 35
        
        for i, (point, hold_id) in enumerate(points_holds_matches.items()):
            cv2.putText(
                frame,
                f"{point}: {hold_id}",
                (x_offset, y_offset + i * line_height),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2
            )
        return frame

    def create_circular_sticker(self, image_path, size=80):
        """Create a circular sticker from an image"""
        # Read the image
        logo = cv2.imread(image_path)
        if logo is None:
            print(f"Error: Could not read image from {image_path}")
            return None
        
        # Resize image
        logo = cv2.resize(logo, (size, size))
        
        # Create a circular mask
        mask = np.zeros((size, size), dtype=np.uint8)
        center = (size // 2, size // 2)
        radius = size // 2
        cv2.circle(mask, center, radius, 255, -1)
        
        # Convert mask to 3 channels
        mask_3channel = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        
        # Apply mask to create circular image
        circular_logo = cv2.bitwise_and(logo, mask_3channel)
        
        # Add slight white border
        border_size = 2
        border_mask = np.zeros((size, size), dtype=np.uint8)
        cv2.circle(border_mask, center, radius, 255, border_size)
        border_mask_3channel = cv2.cvtColor(border_mask, cv2.COLOR_GRAY2BGR)
        
        # Combine logo with border
        circular_logo[border_mask_3channel > 0] = 255
        
        return circular_logo

    def process_frame(self, frame, frame_num):
        # Run inference
        result = self.run_inference(frame, frame_num)[0]
        
        # Update history and get data
        self.keypoints_history = result['keypoints_history']
        keypoint_similarities = result['keypoint_similarities']
        hold_detections = result['hold_detections']
        points_holds_matches = result['points_holds_matches']
        
        # Add points_holds_matches to keypoint_similarities
        keypoint_similarities['points_holds_matches'] = str(points_holds_matches)  # Convert dict to string for CSV storage
        
        # Process keypoints and detections
        keypoints = result['keypoint_outputs']
        detections = sv.Detections.from_inference(keypoints)
        keypoints = sv.KeyPoints.from_inference(keypoints)
        
        # Create annotated frames
        heatmap_detections = keypoints.as_detections(selected_keypoint_indices=[9, 10, 15, 16])
        heatmap_frame = self.heatmap_annotator.annotate(frame.copy(), detections=heatmap_detections)
        
        # Track and annotate
        detections = self.tracker.update_with_detections(detections)
        labels = [f"#{tracker_id} climber" for tracker_id in detections.tracker_id]
        tracked_frame = self.label_annotator.annotate(frame.copy(), detections=detections, labels=labels)
        traced_frame = self.trace_annotator.annotate(tracked_frame, detections=detections)
        
        # Annotate keypoints and holds
        keypoints_frame = self.edge_annotator.annotate(frame.copy(), key_points=keypoints)
        hold_detections = sv.Detections.from_inference(hold_detections)
        keypoints_holds_frame = self.box_annotator.annotate(keypoints_frame, detections=hold_detections)
        
        # Combine frames
        concatenated_frame = cv2.hconcat([traced_frame, keypoints_holds_frame, heatmap_frame])
        
        # Add matches text
        concatenated_frame = self.draw_matches_text(concatenated_frame, points_holds_matches)
        
        # Update and add plots
        plot_image = self.update_plots(frame_num, keypoint_similarities)
        if plot_image is not None:
            h, w = plot_image.shape[:2]
            concatenated_frame[50:50+h, 50:50+w] = plot_image
        
        # Add logo sticker
        if not hasattr(self, 'logo_sticker'):
            logo_path = '/Users/danielreiff/Documents/BoulderVision/assets/u9811453479_httpss.mj.runozM1nDbAHG0_can_you_write_BoulderVis_e4612d8a-df8b-4059-b99a-475dac29071e_3.png'
            self.logo_sticker = self.create_circular_sticker(logo_path, size=120)
        
        if self.logo_sticker is not None:
            sticker_size = self.logo_sticker.shape[0]
            y_offset = concatenated_frame.shape[0] - sticker_size - 20
            x_offset = concatenated_frame.shape[1] - sticker_size - 20
            
            # Create mask for blending
            mask = np.all(self.logo_sticker != [0, 0, 0], axis=2)
            
            # Add sticker to frame
            roi = concatenated_frame[y_offset:y_offset+sticker_size, 
                                   x_offset:x_offset+sticker_size]
            roi[mask] = self.logo_sticker[mask]
        
        return concatenated_frame, keypoint_similarities

def analyze_climb_metrics(csv_path):
    # Read the movement data
    movement_data = pd.read_csv(csv_path)
    
    # Movement Dynamics (velocity_ratio)
    velocity = movement_data['velocity_ratio'].dropna()
    velocity_stats = {
        'min': round(velocity.min(), 2),
        'max': round(velocity.max(), 2),
        'mean': round(velocity.mean(), 2)
    }
    
    # Distance Coverage
    distance = movement_data['cumulative_distance'].dropna()
    distance_stats = {
        'total': round(distance.sum(), 2),  # Sum all frame-to-frame movements
        'mean_per_frame': round(distance.mean(), 2)  # Average movement per frame
    }
    
    # Climb Duration
    holds_data = movement_data['points_holds_matches'].apply(eval)
    has_hold = holds_data.apply(lambda x: 
        x['left_wrist'] is not None or x['right_wrist'] is not None)
    
    start_frame = movement_data[has_hold].iloc[0]['frame_num']
    end_frame = movement_data[has_hold].iloc[-1]['frame_num']
    duration_frames = end_frame - start_frame
    
    return velocity_stats, distance_stats, duration_frames

def main():
    # Load configuration
    with open('config.yaml', 'r') as file:
        config = yaml.safe_load(file)
    
    # Configuration
    API_URL = config['api']['url']
    API_KEY = os.getenv("ROBOFLOW_API_KEY")  # Keep this as env var for security
    WORKSPACE_NAME = config['api']['workspace_name']
    WORKFLOW_ID = config['api']['workflow_id']
    VIDEO_PATH = config['video']['path']
    OUTPUT_VIDEO_PATH = config['output']['video_path']
    OUTPUT_DATA_PATH = config['output']['data_path']
    
    # Setup video processing
    frames_generator, total_frames, video_info = setup_video_processing(
        vid_path=VIDEO_PATH,
        start_seconds=config['video']['start_seconds'],
        end_seconds=config['video']['end_seconds'],
        stride=config['video']['stride']
    )
    
    # Initialize analyzer
    analyzer = ClimbingAnalyzer(API_URL, API_KEY, WORKSPACE_NAME, WORKFLOW_ID)
    
    # Setup video output
    output_video_info = sv.VideoInfo(
        width=int(1080*3),
        height=1920,
        fps=video_info.fps
    )
    
    movement_data = []
    
    with sv.VideoSink(target_path=OUTPUT_VIDEO_PATH, video_info=output_video_info) as sink:
        for frame_num, frame in tqdm(enumerate(frames_generator)):
            processed_frame, keypoint_similarities = analyzer.process_frame(frame, frame_num)
            keypoint_similarities['frame_num'] = frame_num
            movement_data.append(keypoint_similarities)
            sink.write_frame(processed_frame)
    
    # Save movement data
    pd.DataFrame(movement_data).to_csv(OUTPUT_DATA_PATH, index=False)
    
    # After processing is complete
    print("\nProcessing complete. Analyzing climb metrics...")
    
    velocity_stats, distance_stats, duration = analyze_climb_metrics(OUTPUT_DATA_PATH)
    
    print("\nClimbing Movement Analysis:")
    print(f"\nClimb Duration: {duration} frames")
    print(f"Climb Duration: {duration/video_info.fps} seconds")
    
    print("\n1. Movement Dynamics (velocity ratio):")
    print(f"• Slowest movement: {velocity_stats['min']}x average speed")
    print(f"• Fastest movement: {velocity_stats['max']}x average speed")
    print(f"• Average movement speed: {velocity_stats['mean']}x")
    print("  (Values > 1 indicate faster than average movements, < 1 indicate slower movements)")
    
    print("\n2. Distance Coverage (in pixels):")
    print(f"• Total distance covered: {distance_stats['total']}")
    print(f"• Average distance per frame: {distance_stats['mean_per_frame']}")
    print("  (Higher values indicate more dynamic movement, lower values suggest static positions)")

if __name__ == "__main__":
    main()

