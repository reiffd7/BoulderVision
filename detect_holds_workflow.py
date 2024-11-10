from inference import InferencePipeline
import cv2
from utils import setup_video_processing
import matplotlib.pyplot as plt
from inference_sdk import InferenceHTTPClient
import supervision as sv
from tqdm import tqdm

def my_sink(result, video_frame):
    keypoints_predictions = result['pose_keypoints']['predictions']
    hold_detections = result['hold_detections']

    print(hold_detections.xyxy)
    print(hold_detections.confidence)
    print(hold_detections.data['class_name'])
    
    # Only process if there are predictions (array is not empty)
    if len(keypoints_predictions.data['keypoints_xy']) > 0:
        keypoints_data = {
            'keypoint_names': keypoints_predictions.data['keypoints_class_name'][0],  # List of keypoint names
            'keypoint_ids': keypoints_predictions.data['keypoints_class_id'][0],      # List of keypoint IDs
            'keypoint_conf': keypoints_predictions.data['keypoints_confidence'][0],   # Confidence scores
            'keypoint_coords': keypoints_predictions.data['keypoints_xy'][0]          # [x,y] coordinates
        }
        print(keypoints_data)

def process_video(frames_generator, tracker, box_annotator, label_annotator, trace_annotator, client, sink):
    for frame in tqdm(frames_generator):
        result = client.run_workflow(
            workspace_name="daniels-workspace-dnmsg",
            workflow_id="detect-holds",
            images={
                "image": frame
            }
        )
        keypoints = result[0]['keypoint_outputs']
        hold_detections = result[0]['hold_detections']
        print(hold_detections)
        print('- - - -  -')
        height = frame.shape[0]
        width = frame.shape[1]

        ## Annotate frame with tracker and trace
        detections = sv.Detections.from_inference(keypoints)
        detections = tracker.update_with_detections(detections)
        labels = [
            f"#{tracker_id} climber"
            for tracker_id
            in detections.tracker_id
        ]
        tracked_annotated_frame = label_annotator.annotate(
            frame.copy(), detections=detections, labels=labels)
        traced_annotated_frame = trace_annotator.annotate(
        tracked_annotated_frame, detections=detections)
        # traced_annotated_frame_resized = cv2.resize(traced_annotated_frame, (width//2, height))
        
        ## Annotate frame with keypoints

        keypoints = sv.KeyPoints.from_inference(keypoints)
        keypoints_annotated_frame = edge_annotator.annotate(
            scene=frame.copy(),
            key_points=keypoints
        )
        detections = sv.Detections.from_inference(hold_detections)
        keypoints_and_holds_annotated_frame = box_annotator.annotate(
            keypoints_annotated_frame, detections=detections
        )


        # keypoints_annotated_frame_resized = cv2.resize(keypoints_annotated_frame, (width//2, height))


        concatenated_frame = cv2.hconcat([traced_annotated_frame, keypoints_and_holds_annotated_frame])
        print(concatenated_frame.shape)

        sink.write_frame(concatenated_frame)


frames_generator, total_frames, video_info = setup_video_processing(
    vid_path='/users/danielreiff/Downloads/IMG_3841.MOV',
    start_seconds=31,
    end_seconds=32,
    stride=1
)

tracker = sv.ByteTrack()
box_annotator = sv.BoundingBoxAnnotator()
label_annotator = sv.LabelAnnotator()
trace_annotator = sv.TraceAnnotator()
edge_annotator = sv.EdgeAnnotator(
    color=sv.Color.GREEN,
    thickness=5
)


CLIENT = InferenceHTTPClient(
    api_url="http://localhost:9001", # use local inference server
    api_key="CStUD4FGBWGrkAzyK7ou"
)

# result = client.run_workflow(
#     workspace_name="daniels-workspace-dnmsg",
#     workflow_id="detect-holds",
#     images={
#         "image": frame
#     }
# )   

# keypoints = result[0]['keypoint_outputs']
# hold_detections = result[0]['hold_detections']
output_video_info = sv.VideoInfo(
    width=int(1080*2),
    height=1920,
    fps=video_info.fps
)

with sv.VideoSink(target_path='output.mp4', video_info=output_video_info) as sink:
    process_video(
        frames_generator=frames_generator,
        tracker=tracker,
        box_annotator=box_annotator,
        client=CLIENT,
        label_annotator=label_annotator,
        trace_annotator=trace_annotator,
        sink=sink
    )

