import cv2
import time
import sys
import logging
import threading
import json
from queue import Queue, Empty
from vision.vision_engine import VisionEngine, VisionEngineError


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class CameraSystemError(Exception):
    pass


class CameraNotAccessibleError(CameraSystemError):
    pass


class FrameCaptureError(CameraSystemError):
    pass


class CameraSystem:
    def __init__(self, camera_index=0, frame_skip=30):

        self.camera_index = camera_index
        self.frame_skip = frame_skip
        self.counter = 0
        self.cap = None
        self.vision = None

        # Threading (UNCHANGED)
        self.frame_queue = Queue(maxsize=1)
        self.result_queue = Queue(maxsize=10)
        self.running = False
        self.processing_thread = None
        self.latest_result = None
        self.result_lock = threading.Lock()
        self.frames_sent = 0
        self.results_received = 0

        try:
            self.cap = cv2.VideoCapture(camera_index)
            if not self.cap.isOpened():
                raise CameraNotAccessibleError("Camera not accessible")

            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            logger.info(f"✅ Camera {camera_index} opened successfully")

        except cv2.error as e:
            raise CameraNotAccessibleError(str(e))

        try:
            self.vision = VisionEngine()
            logger.info("✅ Vision engine initialized successfully")

        except Exception as e:
            self.cleanup()
            raise VisionEngineError(str(e))

    # =========================
    # 🔥 MULTI-FRAME CAPTURE (FIXED + SAFE)
    # =========================
    def capture_and_analyze(self, num_frames=1):

        if self.cap is None or not self.cap.isOpened():
            raise CameraNotAccessibleError("Camera is not initialized")

        # ✅ Warmup (kept from your code)
        for _ in range(10):
            self.cap.read()
        time.sleep(0.5)

        results = []

        for i in range(num_frames):
            ret, frame = self.cap.read()

            if not ret:
                logger.warning(f"⚠️ Frame {i} capture failed")
                continue

            try:
                frame = cv2.resize(frame, (640, 480))
            except:
                continue

            logger.info(f"📸 Capturing frame {i+1}/{num_frames}")

            try:
                result = self.vision.analyze_frame(frame)

                # ✅ CLEAN JSON (your version + safer)
                try:
                    cleaned = result.strip()

                    if cleaned.startswith("```"):
                        parts = cleaned.split("```")
                        cleaned = parts[1] if len(parts) > 1 else cleaned
                        if cleaned.startswith("json"):
                            cleaned = cleaned[4:]

                    result = json.loads(cleaned.strip())

                except Exception as e:
                    logger.warning(f"⚠️ JSON parse failed: {e}")
                    result = {"raw": result}

                results.append(result)

                # ✅ STOP early (optimization)
                break

            except Exception as e:
                logger.error(f"❌ Vision error on frame {i}: {str(e)}")

        return self._aggregate_results(results)

    # =========================
    # 🔥 AGGREGATION (FIXED)
    # =========================
    def _aggregate_results(self, results):

        if not results:
            return {
                "scene_summary": "No visual data available",
                "people": [],
                "objects": [],
                "actions": [],
                "motion_detected": False,
                "important_elements": []
            }

        aggregated = {
            "scene_summary": [],
            "people": [],
            "objects": [],
            "actions": [],
            "motion_detected": False,
            "important_elements": []
        }

        for res in results:
            aggregated["scene_summary"].append(res.get("scene_summary", ""))

            aggregated["people"].extend(res.get("people", []))
            aggregated["objects"].extend(res.get("objects", []))
            aggregated["actions"].extend(res.get("actions", []))
            aggregated["important_elements"].extend(res.get("important_elements", []))

            if res.get("motion_detected"):
                aggregated["motion_detected"] = True

        # ✅ FIX: safe deduplication (no dict error)
        def deduplicate(items):
            seen = set()
            unique = []
            for item in items:
                key = str(item)
                if key not in seen:
                    seen.add(key)
                    unique.append(item)
            return unique

        aggregated["people"] = deduplicate(aggregated["people"])
        aggregated["objects"] = deduplicate(aggregated["objects"])
        aggregated["actions"] = deduplicate(aggregated["actions"])
        aggregated["important_elements"] = list(set(aggregated["important_elements"]))

        aggregated["scene_summary"] = " | ".join(aggregated["scene_summary"])

        return aggregated

    # =========================
    # EXISTING STREAM (UNCHANGED)
    # =========================
    def _process_frames_worker(self):
        logger.info("🔄 Vision processing thread started")

        while self.running:
            try:
                frame = self.frame_queue.get(timeout=0.5)

                if frame is None:
                    continue

                logger.info("🧠 Processing frame in background...")

                try:
                    result = self.vision.analyze_frame(frame)

                    with self.result_lock:
                        self.latest_result = result
                        self.results_received += 1

                    try:
                        self.result_queue.put(result, block=False)
                    except:
                        pass

                    logger.info("✅ Frame processing complete")

                except Exception as e:
                    logger.error(f"❌ Vision error: {str(e)}")

                self.frame_queue.task_done()

            except Empty:
                continue
            except Exception as e:
                logger.error(f"❌ Thread error: {str(e)}")

    def process_stream(self):
        print("🚀 Camera started... Press 'q' to exit")

        self.running = True
        self.processing_thread = threading.Thread(
            target=self._process_frames_worker, daemon=True
        )
        self.processing_thread.start()

        try:
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    continue

                frame = cv2.resize(frame, (640, 480))

                if self.counter % self.frame_skip == 0:
                    if self.frame_queue.full():
                        try:
                            self.frame_queue.get_nowait()
                        except:
                            pass

                    self.frame_queue.put_nowait(frame.copy())

                self.counter += 1

                cv2.imshow("Camera Feed", frame)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        finally:
            self.cleanup()

    def cleanup(self):
        self.running = False

        if self.processing_thread:
            self.processing_thread.join(timeout=2)

        if self.cap:
            self.cap.release()

        cv2.destroyAllWindows()
        print("🛑 Camera stopped")


if __name__ == "__main__":
    cam = CameraSystem(frame_skip=15)
    cam.process_stream()
