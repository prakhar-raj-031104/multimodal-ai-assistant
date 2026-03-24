import os
import cv2
import base64
import logging
from openai import OpenAI, OpenAIError, APIError, APIConnectionError, RateLimitError, APITimeoutError
from typing import Optional



# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class VisionEngineError(Exception):
    """Base exception for vision engine errors"""
    pass


class APIConfigurationError(VisionEngineError):
    """Raised when API configuration is invalid"""
    pass


class ImageEncodingError(VisionEngineError):
    """Raised when image encoding fails"""
    pass


class APIRequestError(VisionEngineError):
    """Raised when API request fails"""
    pass


class VisionEngine:
    def __init__(self, base_url: str = "https://router.huggingface.co/v1", 
                 model: str = "Qwen/Qwen2.5-VL-7B-Instruct:hyperbolic",
                 max_tokens: int = 800,
                 timeout: float = 30.0):
        """
        Initialize vision engine with Hugging Face API.
        
        Args:
            base_url: API base URL
            model: Vision model to use
            max_tokens: Maximum tokens in response
            timeout: Request timeout in seconds
            
        Raises:
            APIConfigurationError: If API configuration is invalid
        """
        self.model = model
        self.max_tokens = max_tokens
        self.timeout = timeout
        
        # Get API token from environment
        api_key = "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"
        if not api_key:
            raise APIConfigurationError(
                "HF_TOKEN environment variable not set. "
                "Please set your Hugging Face token: export HF_TOKEN='your_token'"
            )
        
        if len(api_key.strip()) == 0:
            raise APIConfigurationError("HF_TOKEN is empty. Please provide a valid token.")
        
        try:
            self.client = OpenAI(
                base_url=base_url,
                api_key=api_key,
                timeout=timeout,
            )
            logger.info(f"✅ Vision engine initialized with model: {model}")
            
        except Exception as e:
            raise APIConfigurationError(
                f"Failed to initialize OpenAI client: {str(e)}"
            ) from e

        self.prompt = """
You are a perception module in a robotic AI system specialized in detecting human actions and movements.

Analyze the image and return STRICT JSON with:

- scene_summary: detailed description of environment and what's happening
- people: list of all people detected with attributes (position, pose, body language, what they're doing)
- objects: list of all visible objects with attributes (name, color, size, position)
- text_in_scene: all readable text (exact words)
- spatial_relations: relationships between objects and people
- actions: **CRITICAL** - detect ALL ongoing actions, movements, and activities including:
  * Body movements (dancing, walking, running, jumping, sitting, standing, gesturing)
  * Hand movements (waving, pointing, holding objects)
  * Facial expressions and head movements
  * Any physical activity or motion
  * Body posture and stance
- motion_detected: true/false - is there ANY movement or action happening
- activity_level: "static", "low", "medium", "high" - rate the amount of movement
- important_elements: key items relevant for decision making

Rules:
- **Pay SPECIAL attention to human actions and movements - this is the most important part**
- Look for ANY signs of motion, activity, or dynamic poses
- Dancing, gesturing, exercising should ALWAYS be detected
- Be precise and exhaustive about movements
- Do NOT hallucinate
- Only include visible information
- Output ONLY valid JSON
"""

    def encode_image(self, frame) -> str:
        """
        Encode frame to base64 JPEG.
        
        Args:
            frame: OpenCV frame (numpy array)
            
        Returns:
            Base64 encoded JPEG string
            
        Raises:
            ImageEncodingError: If encoding fails
        """
        if frame is None:
            raise ImageEncodingError("Frame is None, cannot encode")
        
        if frame.size == 0:
            raise ImageEncodingError("Frame is empty, cannot encode")
        
        try:
            success, buffer = cv2.imencode(".jpg", frame)
            
            if not success:
                raise ImageEncodingError("cv2.imencode() failed to encode frame as JPEG")
            
            if buffer is None or buffer.size == 0:
                raise ImageEncodingError("Encoded buffer is empty")
            
            encoded = base64.b64encode(buffer).decode("utf-8")
            logger.debug(f"Frame encoded successfully (size: {len(encoded)} chars)")
            return encoded
            
        except cv2.error as e:
            raise ImageEncodingError(f"OpenCV error during encoding: {str(e)}") from e
        except Exception as e:
            raise ImageEncodingError(f"Unexpected error during encoding: {str(e)}") from e

    def analyze_frame(self, frame) -> str:
        """
        Analyze frame using vision AI model.
        
        Args:
            frame: OpenCV frame (numpy array)
            
        Returns:
            JSON string with analysis results
            
        Raises:
            ImageEncodingError: If frame encoding fails
            APIRequestError: If API request fails
        """
        # Encode image
        try:
            base64_img = self.encode_image(frame)
        except ImageEncodingError:
            raise  # Re-raise with original context
        except Exception as e:
            raise ImageEncodingError(f"Unexpected error during image encoding: {str(e)}") from e
        
        # Make API request
        try:
            logger.debug(f"Sending request to {self.model}...")
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": self.prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{base64_img}"
                                },
                            },
                        ],
                    }
                ],
                max_tokens=self.max_tokens,
            )

            if not response or not response.choices:
                raise APIRequestError("API returned empty response")
            
            content = response.choices[0].message.content
            if content is None:
                raise APIRequestError("API returned None content")
            
            logger.debug("✅ Frame analysis completed successfully")
            return content

        except APITimeoutError as e:
            raise APIRequestError(
                f"API request timed out after {self.timeout}s: {str(e)}"
            ) from e
            
        except APIConnectionError as e:
            raise APIRequestError(
                f"Failed to connect to API. Check your internet connection: {str(e)}"
            ) from e
            
        except RateLimitError as e:
            raise APIRequestError(
                f"API rate limit exceeded. Please wait before retrying: {str(e)}"
            ) from e
            
        except APIError as e:
            raise APIRequestError(
                f"API error occurred: {str(e)}"
            ) from e
            
        except OpenAIError as e:
            raise APIRequestError(
                f"OpenAI client error: {str(e)}"
            ) from e
            
        except Exception as e:
            raise APIRequestError(
                f"Unexpected error during API request: {str(e)}"
            ) from e