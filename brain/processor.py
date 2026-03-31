from typing import Optional, Dict
from pathlib import Path
from brain.context_builder import build_context
from brain.prompt_manager import build_prompt
from brain.llm_engine import generate_response
     

def process_user_query(
    user_input: str,
    conversation_history: Optional[str] = None,
    vision_data: Optional[Dict] = None,
    memory_data: Optional[str] = None,
    instruction: Optional[str] = None,
    audio_data: Optional[str] = None,
    audio_file_path: Optional[str] = None,
) -> dict:
    """
    Core brain pipeline (pure reasoning layer)
    Main function to process user query through the AI pipeline.

    Audio can be provided as pre-transcribed text (`audio_data`) or as a local
    file path (`audio_file_path`) that will be transcribed using Voxtral.
    """
    
    if not audio_data and audio_file_path:
        resolved_path = str(Path(audio_file_path).expanduser())
        audio_data = transcribe_audio_file(resolved_path)

    # Step 1: Build context
    context = build_context(
        user_input=user_input,
        conversation_history=conversation_history,
        vision_data=vision_data,
        audio_data=audio_data,
        memory_data=memory_data
    )

    # Step 2: Build prompt
    prompt = build_prompt(context, instruction)

    # Step 3: Generate response
    response = generate_response(prompt)

    return {
        "response": response,
        "context_used": context  ,
        "audio_transcript": audio_data,
    }