import cv2
import time
import sys
import logging
import threading
from queue import Queue, Empty
from vision_engine import VisionEngine, VisionEngineError


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class CameraSystemError(Exception):
    """Base exception for camera system errors"""
    pass


class CameraNotAccessibleError(CameraSystemError):
    """Raised when camera cannot be opened"""
    pass


class FrameCaptureError(CameraSystemError):
    """Raised when frame capture fails"""
    pass


class CameraSystem:
    def __init__(self, camera_index=0, frame_skip=30):
        """
        Initialize camera system with threaded processing.
        
        Args:
            camera_index: Camera device index (default: 0)
            frame_skip: Process every Nth frame (default: 30)
            
        Raises:
            CameraNotAccessibleError: If camera cannot be opened
            VisionEngineError: If vision engine initialization fails
        """
        self.camera_index = camera_index
        self.frame_skip = frame_skip
        self.counter = 0
        self.cap = None
        self.vision = None
        
        # Threading components
        self.frame_queue = Queue(maxsize=1)  # Only keep latest frame to avoid backlog
        self.result_queue = Queue(maxsize=10)  # Buffer for results
        self.running = False
        self.processing_thread = None
        self.latest_result = None
        self.result_lock = threading.Lock()
        self.frames_sent = 0
        self.results_received = 0
        
        try:
            self.cap = cv2.VideoCapture(camera_index)
            if not self.cap.isOpened():
                raise CameraNotAccessibleError(
                    f"❌ Camera at index {camera_index} not accessible. "
                    f"Please check if camera is connected and not in use by another application."
                )
            
            # Set camera buffer size to 1 to reduce latency
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            
            logger.info(f"✅ Camera {camera_index} opened successfully")
            
        except cv2.error as e:
            raise CameraNotAccessibleError(
                f"❌ OpenCV error while accessing camera {camera_index}: {str(e)}"
            ) from e
            
        try:
            self.vision = VisionEngine()
            logger.info("✅ Vision engine initialized successfully")
            
        except VisionEngineError as e:
            self.cleanup()
            raise VisionEngineError(f"❌ Failed to initialize vision engine: {str(e)}") from e
        except Exception as e:
            self.cleanup()
            raise VisionEngineError(
                f"❌ Unexpected error during vision engine initialization: {str(e)}"
            ) from e
    
    def _process_frames_worker(self):
        """Worker thread that processes frames asynchronously"""
        logger.info("🔄 Vision processing thread started")
        
        while self.running:
            try:
                # Get frame from queue with timeout
                frame = self.frame_queue.get(timeout=0.5)
                
                if frame is None:
                    continue
                
                logger.info("🧠 Processing frame in background...")
                
                try:
                    result = self.vision.analyze_frame(frame)
                    
                    # Update latest result
                    with self.result_lock:
                        self.latest_result = result
                        self.results_received += 1
                    
                    # Put result in queue for main thread to print
                    try:
                        self.result_queue.put(result, block=False)
                    except:
                        # If queue is full, remove oldest and add new
                        try:
                            self.result_queue.get_nowait()
                            self.result_queue.put(result, block=False)
                        except:
                            pass
                    
                    logger.info("✅ Frame processing complete")
                    
                except VisionEngineError as e:
                    logger.error(f"❌ Vision analysis error: {str(e)}")
                    # Put error in result queue so main thread knows
                    try:
                        self.result_queue.put(f'{{"error": "Vision analysis failed: {str(e)}"}}', block=False)
                    except:
                        pass
                except Exception as e:
                    logger.error(f"❌ Unexpected error during vision analysis: {str(e)}")
                    try:
                        self.result_queue.put(f'{{"error": "Unexpected error: {str(e)}"}}', block=False)
                    except:
                        pass
                
                self.frame_queue.task_done()
                
            except Empty:
                continue
            except Exception as e:
                logger.error(f"❌ Error in processing thread: {str(e)}")
        
        logger.info("🛑 Vision processing thread stopped")

    def process_stream(self):
        """
        Start processing camera stream with async vision processing.
        
        Raises:
            FrameCaptureError: If frame capture consistently fails
            KeyboardInterrupt: If user interrupts with Ctrl+C
        """
        print("🚀 Camera started... Press 'q' to exit or Ctrl+C to stop")
        print("📹 Camera feed runs independently from AI processing (no freezing!)\n")
        
        # Start background processing thread
        self.running = True
        self.processing_thread = threading.Thread(target=self._process_frames_worker, daemon=True)
        self.processing_thread.start()
        
        consecutive_failures = 0
        max_consecutive_failures = 5
        last_print_time = 0

        try:
            while True:
                try:
                    # Read frame - this should NEVER block for long
                    ret, frame = self.cap.read()
                    if not ret:
                        consecutive_failures += 1
                        logger.warning(
                            f"⚠️ Frame capture failed (attempt {consecutive_failures}/{max_consecutive_failures})"
                        )
                        
                        if consecutive_failures >= max_consecutive_failures:
                            raise FrameCaptureError(
                                f"❌ Failed to capture frame {max_consecutive_failures} times consecutively. "
                                f"Camera may have been disconnected."
                            )
                        
                        time.sleep(0.1)  # Brief pause before retry
                        continue
                    
                    # Reset failure counter on successful capture
                    consecutive_failures = 0

                    try:
                        frame = cv2.resize(frame, (640, 480))
                    except cv2.error as e:
                        logger.error(f"❌ Error resizing frame: {str(e)}")
                        continue

                    # Send frame to processing thread (non-blocking)
                    if self.counter % self.frame_skip == 0:
                        # Try to add to queue without blocking
                        try:
                            # If queue is full, clear it and add new frame (always process latest)
                            if self.frame_queue.full():
                                try:
                                    self.frame_queue.get_nowait()
                                    logger.debug("⚠️ Dropped old frame, API too slow")
                                except:
                                    pass
                            
                            self.frame_queue.put_nowait(frame.copy())
                            self.frames_sent += 1
                            logger.debug(f"📤 Frame sent to processing queue (sent: {self.frames_sent}, received: {self.results_received})")
                        except:
                            logger.debug("⚠️ Processing queue full, skipping frame")

                    self.counter += 1

                    # Check for results and print ALL available results
                    results_printed = 0
                    while results_printed < 3:  # Limit to prevent blocking too long
                        try:
                            result = self.result_queue.get_nowait()
                            print("\n========== VISION OUTPUT ==========")
                            print(result)
                            print("===================================\n")
                            results_printed += 1
                        except Empty:
                            break  # No more results

                    try:
                        cv2.imshow("Camera Feed", frame)
                    except cv2.error as e:
                        logger.error(f"❌ Error displaying frame: {str(e)}")
                        # Continue even if display fails

                    # Non-blocking key check
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        logger.info("User pressed 'q' to exit")
                        break
                        
                except KeyboardInterrupt:
                    raise  # Re-raise to outer handler
                except FrameCaptureError:
                    raise  # Re-raise to outer handler
                except Exception as e:
                    logger.error(f"❌ Unexpected error in frame processing loop: {str(e)}")
                    print(f"⚠️ Error occurred but continuing: {str(e)}\n")
                    time.sleep(0.1)  # Brief pause before continuing

        except KeyboardInterrupt:
            logger.info("\n⚠️ Interrupted by user (Ctrl+C)")
            print("\n⚠️ Stopping camera system...")
        except FrameCaptureError as e:
            logger.error(str(e))
            print(f"\n{str(e)}")
        except Exception as e:
            logger.error(f"❌ Fatal error in process_stream: {str(e)}")
            print(f"\n❌ Fatal error: {str(e)}")
        finally:
            self.cleanup()

    def cleanup(self):
        """Release camera resources and close windows"""
        # Stop processing thread
        self.running = False
        if self.processing_thread is not None:
            logger.info("Waiting for processing thread to finish...")
            self.processing_thread.join(timeout=2.0)
            
        try:
            if self.cap is not None:
                self.cap.release()
                logger.info("Camera released")
        except Exception as e:
            logger.error(f"Error releasing camera: {str(e)}")
            
        try:
            cv2.destroyAllWindows()
            logger.info("Windows closed")
        except Exception as e:
            logger.error(f"Error closing windows: {str(e)}")
            
        print("🛑 Camera stopped")


if __name__ == "__main__":
    try:
        cam = CameraSystem(frame_skip=15)  # Process every 15th frame for faster detection
        cam.process_stream()
        sys.exit(0)
        
    except CameraNotAccessibleError as e:
        logger.error(str(e))
        print(f"\n{str(e)}")
        print("\nTroubleshooting tips:")
        print("1. Check if camera is physically connected")
        print("2. Try a different camera index (0, 1, 2, etc.)")
        print("3. Close other applications using the camera")
        print("4. Check camera permissions")
        sys.exit(1)
        
    except VisionEngineError as e:
        logger.error(str(e))
        print(f"\n{str(e)}")
        print("\nTroubleshooting tips:")
        print("1. Check if HF_TOKEN environment variable is set")
        print("2. Verify your Hugging Face token is valid")
        print("3. Check internet connectivity")
        sys.exit(1)
        
    except Exception as e:
        logger.error(f"Unexpected fatal error: {str(e)}", exc_info=True)
        print(f"\n❌ Unexpected fatal error: {str(e)}")
        print("Check logs for more details")
        sys.exit(1)