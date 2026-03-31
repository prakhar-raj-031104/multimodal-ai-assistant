from brain.processor import process_user_query
from vision.vision_adapter import VisionAdapter
from speech.speech_engine import SpeechEngine
from memory.memory_manager import MemoryManager
from backend.request_wrapper import MultimodalRequest


class Assistant:

    def __init__(self):
        self.vision = VisionAdapter()
        self.speech = SpeechEngine()
        self.memory = MemoryManager()

    def run(self, user_input: str):

        print("🚀 Running Full Multimodal Pipeline...\n")

        # Step 1: Vision
        print("📸 Getting vision input...")
        vision_data = self.vision.get_input()

        # Step 2: Audio
        print("🎤 Getting audio input...")
        audio_data = self.speech.transcribe()

        # Step 3: Memory (RAG)
        print("🧠 Retrieving memory...")
        query = user_input or str(vision_data) or audio_data
        memory_data = self.memory.retrieve(query)

        # ✅ FIXED HERE (use audio_data consistently)
        request = MultimodalRequest(
            user_input=user_input or audio_data,
            vision_data=vision_data,
            audio_data=audio_data,
            memory_data=memory_data
        )

        # ⚡ PROCESS
        result = process_user_query(
            user_input=request.user_input,
            vision_data=request.vision_data,
            audio_data=request.audio_data,
            memory_data=request.memory_data
        )
        
        return result
