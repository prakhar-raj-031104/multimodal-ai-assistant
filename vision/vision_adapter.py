from vision.camera import CameraSystem

class VisionAdapter:
    def __init__(self):
        self.camera = CameraSystem()

    def get_input(self):
        return self.camera.capture_and_analyze()  # you added this earlier