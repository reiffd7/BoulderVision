import supervision as sv


def setup_video_processing(
    vid_path: str,
    start_seconds: float = 0,
    end_seconds: float = None,
    stride: int = 1
) -> tuple:
    """
    Setup video processing with time-based controls
    
    Args:
        vid_path: Path to video file
        start_seconds: Start time in seconds (default: 0 = start of video)
        end_seconds: End time in seconds (default: None = end of video)
        stride: Process every nth frame (default: 1 = process all frames)
    
    Returns:
        tuple: (frames_generator, total_frames, video_info)
    """
    video_info = sv.VideoInfo.from_video_path(vid_path)
    fps = video_info.fps
    
    # Convert seconds to frame numbers
    start_frame = int(start_seconds * fps)
    if end_seconds is None:
        end_frame = video_info.total_frames
        end_seconds = video_info.total_frames / fps
    else:
        end_frame = min(int(end_seconds * fps), video_info.total_frames)
    
    # Calculate actual frames to process
    total_frames = (end_frame - start_frame) // stride
    
    frames_generator = sv.get_video_frames_generator(
        source_path=vid_path,
        start=start_frame,
        end=end_frame,
        stride=stride
    )
    
    total_duration = video_info.total_frames / fps
    print(f"Video setup:")
    print(f"- Duration: {total_duration:.1f}s ({total_duration/60:.1f}min)")
    print(f"- Processing from: {start_seconds:.1f}s ({start_seconds/60:.1f}min)")
    print(f"- Processing to: {end_seconds:.1f}s ({end_seconds/60:.1f}min)")
    print(f"- Frame stride: {stride} (effective FPS: {fps/stride:.1f})")
    print(f"- Frames to process: {total_frames}")
    
    return frames_generator, total_frames, video_info