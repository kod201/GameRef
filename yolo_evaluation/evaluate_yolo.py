# evaluate_yolo.py
import argparse
import json
import logging
import os
import sys
import time
import platform # Added
import yaml
from pathlib import Path

import cv2
import torch
# Ensure ultralytics is installed: pip install --upgrade ultralytics torch torchvision opencv-python pyyaml
try:
    from ultralytics import YOLO
    # Note: Metrics access might differ slightly in newer ultralytics versions
    # from ultralytics.utils.metrics import ConfusionMatrix # May not be needed directly
    # from ultralytics.utils.checks import check_requirements # May not be needed directly
except ImportError:
    print("Error: 'ultralytics' package not found. Please install it: pip install --upgrade ultralytics torch torchvision opencv-python pyyaml")
    sys.exit(1)

# Configure logging to stderr
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s', stream=sys.stderr)

def parse_args():
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(description="Evaluate YOLOv8-tiny model throughput and accuracy using a camera feed.") # Updated description
    parser.add_argument("MODEL_WEIGHTS", type=str, help="Path or URL to YOLOv8-tiny weights (e.g., yolov8n.pt).")
    # Removed IMAGES_DIR, added camera/frame args
    parser.add_argument("--CAM_ID", type=int, default=0, help="Integer index of the camera (default: 0).")
    parser.add_argument("--NUM_FRAMES", type=int, default=300, help="Number of consecutive frames to benchmark (default: 300).")
    parser.add_argument("--SAVE_DIR", type=str, default=None,
                        help="(Optional) Directory to save captured frames. Required for accuracy calculation.")
    parser.add_argument("--LABELS_DIR", type=str, default=None,
                        help="(Optional) Directory containing YOLO-format .txt labels for saved frames. Requires SAVE_DIR.")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size for validation (default: 640).")
    parser.add_argument("--batch", type=int, default=8, help="Batch size for validation (default: 8).")
    parser.add_argument("--output_file", type=str, default="yolov8_tiny_camera_eval_report.json", # Updated default filename
                        help="Path to save the JSON report (default: yolov8_tiny_camera_eval_report.json).")
    parser.add_argument("--frame_width", type=int, default=None, help="Optional camera frame width.") # Added
    parser.add_argument("--frame_height", type=int, default=None, help="Optional camera frame height.") # Added

    return parser.parse_args()

def validate_args(args):
    """Validates the parsed arguments."""
    # Updated validation logic for SAVE_DIR and LABELS_DIR interaction
    if args.SAVE_DIR:
        args.SAVE_DIR = Path(args.SAVE_DIR).resolve()
        try:
            args.SAVE_DIR.mkdir(parents=True, exist_ok=True)
            logging.info(f"Saving captured frames to: {args.SAVE_DIR}")
        except OSError as e:
            logging.error(f"Could not create SAVE_DIR {args.SAVE_DIR}: {e}")
            sys.exit(1)
    elif args.LABELS_DIR:
         logging.warning("LABELS_DIR provided without SAVE_DIR. Accuracy cannot be calculated without saved frames.")
         args.LABELS_DIR = None # Ignore labels if not saving frames

    if args.LABELS_DIR:
        args.LABELS_DIR = Path(args.LABELS_DIR).resolve()
        if not args.LABELS_DIR.is_dir():
            logging.warning(f"LABELS_DIR provided but not found: {args.LABELS_DIR}. Accuracy will not be calculated.")
            args.LABELS_DIR = None
        elif not args.SAVE_DIR:
             logging.warning("LABELS_DIR provided but SAVE_DIR is not. Accuracy requires saved frames.")
             args.LABELS_DIR = None
    else:
        if args.SAVE_DIR:
            logging.info("SAVE_DIR provided, but LABELS_DIR is not. Only FPS will be calculated.")
        else:
            logging.info("Neither SAVE_DIR nor LABELS_DIR provided. Only FPS will be calculated.")


    # Output file path validation (same as before)
    args.output_file = Path(args.output_file).resolve()
    try:
        args.output_file.parent.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        logging.error(f"Could not create output directory {args.output_file.parent}: {e}")
        sys.exit(1)

# Reusing the label info function from the previous version
def get_label_info(labels_dir):
    """
    Attempts to infer number of classes and names from label files.
    Assumes class indices start from 0.
    Returns nc (int) and names (list of str).
    If inference fails, returns default COCO values as a fallback.
    """
    # ... existing code ...
    # (No changes needed in this function itself, just how it's called)
    nc = 0
    if not labels_dir:
        logging.warning("Cannot infer class info without label directory. Falling back to COCO defaults (80 classes).")
        # Fallback to COCO default if no labels provided for accuracy check
        return 80, [f'class_{i}' for i in range(80)] # Placeholder names

    label_files = list(labels_dir.glob('*.txt'))
    if not label_files:
         logging.warning(f"No label files (.txt) found in {labels_dir}. Falling back to COCO defaults (80 classes).")
         return 80, [f'class_{i}' for i in range(80)]

    max_class_id = -1
    for label_file in label_files:
        try:
            with open(label_file, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 1:
                        class_id = int(parts[0])
                        max_class_id = max(max_class_id, class_id)
        except Exception as e:
            logging.warning(f"Could not read or parse label file {label_file}: {e}")
            continue # Skip problematic files

    if max_class_id == -1:
        logging.warning(f"No valid class IDs found in label files in {labels_dir}. Falling back to COCO defaults (80 classes).")
        return 80, [f'class_{i}' for i in range(80)]

    nc = max_class_id + 1
    names = [f'class_{i}' for i in range(nc)] # Generate generic names
    logging.info(f"Inferred {nc} classes from label files in {labels_dir}.")
    return nc, names

def try_init_camera(cam_id, frame_width=None, frame_height=None):
    """
    Attempt to initialize camera with multiple backends and settings.
    Returns (cap, actual_width, actual_height) or raises an exception with details.
    """
    # List of backends to try
    backends = [
        (cv2.CAP_AVFOUNDATION, "AVFoundation (macOS native)"),  # Try macOS native first
        (cv2.CAP_ANY, "Default"),  # Then try default
    ]
    
    last_error = None
    for backend, backend_name in backends:
        try:
            logging.info(f"Attempting to open camera {cam_id} with {backend_name} backend...")
            cap = cv2.VideoCapture(cam_id + backend)
            
            if not cap.isOpened():
                logging.warning(f"Failed to open camera with {backend_name} backend")
                continue
                
            # Try setting resolution if specified
            if frame_width:
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
            if frame_height:
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)
            
            # Verify camera is working by capturing a test frame
            for _ in range(5):  # Try up to 5 times
                ret, frame = cap.read()
                if ret and frame is not None:
                    actual_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
                    actual_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
                    fps = cap.get(cv2.CAP_PROP_FPS)
                    logging.info(f"Successfully initialized camera:")
                    logging.info(f"- Backend: {backend_name}")
                    logging.info(f"- Resolution: {int(actual_width)}x{int(actual_height)}")
                    logging.info(f"- FPS: {fps}")
                    return cap, actual_width, actual_height
                time.sleep(0.1)  # Small delay between attempts
                
            raise RuntimeError("Camera opened but failed to capture frames")
            
        except Exception as e:
            last_error = f"Error with {backend_name} backend: {str(e)}"
            logging.warning(last_error)
            if cap is not None:
                cap.release()
                
    raise RuntimeError(f"Failed to initialize camera with any backend. Last error: {last_error}")

def main():
    args = parse_args()
    validate_args(args)

    # Updated variable names for clarity
    model_weights = args.MODEL_WEIGHTS
    cam_id = args.CAM_ID
    num_frames_target = args.NUM_FRAMES
    save_dir = args.SAVE_DIR
    labels_dir = args.LABELS_DIR # Already validated
    output_file = args.output_file
    temp_yaml_path = Path("./temp_camera_data.yaml").resolve() # Changed temp filename

    report = {
        "frames_tested": 0,
        "device": None,
        "avg_fps": None,
        "accuracy": None,
        "system": platform.platform() # Added system info
    }

    cap = None # Initialize camera capture object
    cv2.namedWindow('YOLOv8 Detections', cv2.WINDOW_NORMAL)  # Create resizable window
    start_time_total = time.perf_counter()

    try:
        # 1. Load the model (same as before)
        logging.info(f"Loading model from: {model_weights}")
        model = YOLO(model_weights)
        model.fuse()
        device = str(model.device)
        report["device"] = device
        logging.info(f"Model loaded successfully. Benchmarking on device: {device}")

        # 2. Initialize camera with new helper function
        try:
            cap, actual_width, actual_height = try_init_camera(
                cam_id,
                args.frame_width,
                args.frame_height
            )
        except Exception as e:
            logging.error(f"Failed to initialize camera: {e}")
            if cap:
                cap.release()
            cv2.destroyAllWindows()
            sys.exit(1)

        # 3. Warm-up with retry logic
        logging.info("Performing warm-up capture and inference...")
        warm_up_success = False
        for warm_up_attempt in range(3):  # Try up to 3 times
            ret, frame = cap.read()
            if ret and frame is not None:
                try:
                    _ = model(frame, verbose=False)
                    warm_up_success = True
                    logging.info("Warm-up complete.")
                    break
                except Exception as e:
                    logging.warning(f"Warm-up inference failed on attempt {warm_up_attempt + 1}: {e}")
            else:
                logging.warning(f"Failed to capture warm-up frame on attempt {warm_up_attempt + 1}")
            time.sleep(0.5)  # Wait before retry
            
        if not warm_up_success:
            logging.error("Failed to complete warm-up after multiple attempts")
            cap.release()
            cv2.destroyAllWindows()
            sys.exit(1)

        # 4. Benchmark FPS and optionally save frames (New loop logic)
        logging.info(f"Starting benchmark for {num_frames_target} frames...")
        frames_processed = 0
        saved_paths = [] # Keep track of saved image paths if save_dir is set
        inference_start_time = time.perf_counter()
        fps_update_interval = 30  # Update FPS display every 30 frames
        frame_times = []

        while frames_processed < num_frames_target:
            ret, frame = cap.read()
            if not ret or frame is None:
                logging.warning(f"Failed to capture frame {frames_processed + 1}. Stopping benchmark early.")
                break

            # Run inference and get results
            t0 = time.perf_counter()
            results = model(frame, verbose=False)  # Get detection results
            t1 = time.perf_counter()
            frame_times.append(t1 - t0)

            # Draw detection results on frame
            annotated_frame = results[0].plot()  # Plot results on image

            # Calculate and display FPS
            if len(frame_times) > fps_update_interval:
                current_fps = fps_update_interval / sum(frame_times[-fps_update_interval:])
                cv2.putText(annotated_frame, f"FPS: {current_fps:.1f}", (10, 30), 
                          cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # Display the frame
            cv2.imshow('YOLOv8 Detections', annotated_frame)

            # Save frame if requested
            if save_dir:
                try:
                    img_path_str = str(save_dir / f"{frames_processed:06d}.jpg")
                    cv2.imwrite(img_path_str, frame)
                    saved_paths.append(img_path_str)
                except Exception as e:
                     logging.warning(f"Failed to save frame {frames_processed}: {e}")

            frames_processed += 1

            # Break loop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                logging.info("User requested to quit ('q' pressed)")
                break

        inference_end_time = time.perf_counter()
        cv2.destroyAllWindows()  # Close display window
        cap.release() # Release the camera
        logging.info("Camera released.")

        report["frames_tested"] = frames_processed
        if frames_processed == 0:
            logging.error("No frames were processed.")
            sys.exit(1)

        total_inference_time = inference_end_time - inference_start_time
        avg_fps = frames_processed / total_inference_time if total_inference_time > 0 else 0
        report["avg_fps"] = round(avg_fps, 2)
        logging.info(f"Benchmark complete. Processed {frames_processed} frames.")
        logging.info(f"Average throughput: {report['avg_fps']:.2f} FPS")


        # 5. (Optional) Compute Accuracy (Adapted logic)
        metrics = None
        if labels_dir and save_dir and saved_paths: # Check all conditions
            logging.info("Calculating accuracy using saved frames and provided labels...")
            # Ensure labels_dir actually exists (double check after arg parsing logic)
            if not labels_dir.is_dir():
                 logging.warning(f"Labels directory {labels_dir} not found. Skipping accuracy calculation.")
            else:
                # Infer nc and names from labels
                nc, names = get_label_info(labels_dir)

                # Create minimal data.yaml pointing to the SAVE_DIR
                data_yaml_content = {
                    'path': str(save_dir.parent), # Root path for val
                    'val': str(save_dir.relative_to(save_dir.parent)), # Relative path to saved images
                    'names': names,
                    'nc': nc
                }
                try:
                    with open(temp_yaml_path, 'w') as f:
                        yaml.dump(data_yaml_content, f, default_flow_style=False)
                    logging.info(f"Created temporary data configuration for validation: {temp_yaml_path}")

                    # Run validation using the saved frames
                    # Important: Assumes label files in LABELS_DIR have corresponding names
                    # (e.g., 000000.jpg in SAVE_DIR needs 000000.txt in LABELS_DIR)
                    metrics_results = model.val(data=str(temp_yaml_path),
                                        split='val',
                                        imgsz=args.imgsz,
                                        batch=args.batch,
                                        save_json=False,
                                        save_conf=False,
                                        save_txt=False,
                                        plots=False,
                                        verbose=False
                                        )

                    # Extract metrics (adjust based on actual ultralytics version's return object)
                    if hasattr(metrics_results, 'box'):
                        report["accuracy"] = {
                            "mAP50": round(metrics_results.box.map50, 4),
                            "mAP50-95": round(metrics_results.box.map, 4),
                            "precision": round(metrics_results.box.p[0], 4) if metrics_results.box.p.size > 0 else 0.0,
                            "recall": round(metrics_results.box.r[0], 4) if metrics_results.box.r.size > 0 else 0.0,
                        }
                        logging.info(f"Accuracy metrics: mAP50={report['accuracy']['mAP50']}, mAP50-95={report['accuracy']['mAP50-95']}")
                    else:
                         logging.warning("Could not find expected 'box' attribute in validation results. Accuracy metrics unavailable.")
                         report["accuracy"] = None


                except Exception as e:
                    logging.error(f"Error during accuracy validation: {e}", exc_info=True)
                    report["accuracy"] = None # Set to null on error
                finally:
                    # Clean up temporary yaml file
                    if temp_yaml_path.exists():
                        try:
                            temp_yaml_path.unlink()
                            logging.info(f"Removed temporary data configuration: {temp_yaml_path}")
                        except OSError as e:
                            logging.warning(f"Could not remove temporary file {temp_yaml_path}: {e}")
        elif labels_dir and not save_dir:
             logging.warning("Labels provided but frames were not saved. Skipping accuracy.")
        elif save_dir and not labels_dir:
             logging.info("Frames saved but no labels provided. Skipping accuracy.")
        else:
            logging.info("Accuracy calculation not requested or not possible.")
            report["accuracy"] = None


        # 6. Generate and Output Report (Minor changes)
        report_json = json.dumps(report, indent=2)

        # Print to stdout
        print(report_json)

        # Save to file
        try:
            with open(output_file, 'w') as f:
                f.write(report_json)
            logging.info(f"Report saved to: {output_file}")
        except IOError as e:
            logging.error(f"Failed to save report to {output_file}: {e}")
            # Continue execution but log the error

    except Exception as e:
        logging.error(f"An unexpected critical error occurred: {e}", exc_info=True)
        if cap and cap.isOpened():
            cap.release() # Ensure camera is released on error
        sys.exit(1) # Exit with error code for critical failures

    finally:
        # Ensure camera is released if loop finished normally but error occurred after
        if cap and cap.isOpened():
            cap.release()
        # Clean up temp yaml if it still exists
        if temp_yaml_path.exists():
             try:
                 temp_yaml_path.unlink()
             except OSError:
                 pass # Ignore cleanup error if main logic failed

    end_time_total = time.perf_counter()
    total_runtime = end_time_total - start_time_total
    logging.info(f"Script finished successfully in {total_runtime:.2f} seconds.")
    sys.exit(0) # Explicitly exit with success code

if __name__ == "__main__":
    # Updated Example Usage
    # Basic FPS test:
    # python evaluate_yolo.py yolov8n.pt --NUM_FRAMES 100
    #
    # FPS test + Save frames:
    # python evaluate_yolo.py yolov8n.pt --NUM_FRAMES 500 --SAVE_DIR ./captured_frames
    #
    # FPS + Save frames + Accuracy:
    # python evaluate_yolo.py yolov8n.pt --NUM_FRAMES 500 --SAVE_DIR ./captured_frames --LABELS_DIR ./corresponding_labels
    #
    # Specify camera and resolution:
    # python evaluate_yolo.py yolov8n.pt --CAM_ID 1 --frame_width 1280 --frame_height 720
    main()
