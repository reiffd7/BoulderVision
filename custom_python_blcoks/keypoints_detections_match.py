def get_keypoint_colors(keypoints_xy, bboxes, class_names):
    """
    Determine the color for wrist and ankle keypoints based on which bounding box they fall within
    """
    keypoint_colors = {}
    points_of_interest = {
        9: 'left_wrist',
        10: 'right_wrist',
        15: 'left_ankle',
        16: 'right_ankle'
    }
    
    for idx, name in points_of_interest.items():
        kp = keypoints_xy[idx]
        x, y = kp
        color = None
        
        for bbox, class_name in zip(bboxes, class_names):
            x1, y1, x2, y2 = bbox
            if x1 <= x <= x2 and y1 <= y <= y2:
                color = class_name
                break
                
        keypoint_colors[name] = color
        
    return keypoint_colors

def run(self, keypoints, detections) -> BlockResult:
    # Default response for empty keypoints
    points_holds_matches = {
        'left_wrist': None,
        'right_wrist': None,
        'left_ankle': None,
        'right_ankle': None
    }
    
    # Check if keypoints array exists and is not empty
    if (keypoints.data['keypoints_xy'] is not None and 
        keypoints.data['keypoints_xy'].size > 0):
        points_holds_matches = get_keypoint_colors(
            keypoints.data['keypoints_xy'][0],
            detections.xyxy,
            detections.data['class_name']
        )
    
    return {'points_holds_matches': points_holds_matches}