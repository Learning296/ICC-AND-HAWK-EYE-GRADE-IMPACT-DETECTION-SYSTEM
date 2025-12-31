from ultraedge_audio_detector import UltraEdgeDetector

detector = UltraEdgeDetector(sensitivity='medium', frame_rate=30)

results = detector.detect_impacts(
    video_path='test/test.mp4',
    output_video='outputs/annotated_new_test_1.mp4',
    output_plot='outputs/analysis_1.png'
)

print(f"Found {len(results['detections'])} impacts!")