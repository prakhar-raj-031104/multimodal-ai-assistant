class MemoryManager:
    def __init__(self):
        self.memory = [
            "User often asks about scene understanding.",
            "User is interested in symbolic interpretation of images."
        ]

    def retrieve(self, query: str):
        """
        Mock retrieval (simulates RAG)
        """
        return " | ".join(self.memory)

    def store(self, text: str):
        self.memory.append(text)