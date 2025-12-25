import logging
from datetime import datetime
import os

# Configure logging to write to prompts.md
def setup_logging():
    # Create a custom logging format for prompts.md
    class PromptHandler(logging.Handler):
        def emit(self, record):
            log_entry = self.format(record)
            with open("prompts.md", "a", encoding="utf-8") as f:
                f.write(f"\n## {record.levelname}: {datetime.now().isoformat()}\n")
                f.write(f"**Message**: {record.getMessage()}\n")
                if hasattr(record, 'extra_data'):
                    f.write(f"**Extra**: {record.extra_data}\n")
                f.write("---\n")
    
    # Create and configure the custom handler
    prompt_handler = PromptHandler()
    prompt_handler.setLevel(logging.INFO)
    
    # Create a logger for the application
    logger = logging.getLogger("rag_chatbot")
    logger.setLevel(logging.INFO)
    
    # Add the custom handler to the logger
    logger.addHandler(prompt_handler)
    
    # Prevent propagation to root logger to avoid duplicate logs
    logger.propagate = False
    
    return logger

# Initialize the logger
logger = setup_logging()

def log_prompt_response(prompt: str, response: str, session_id: str = None):
    """Log a prompt and response pair to prompts.md"""
    extra_data = {"session_id": session_id} if session_id else {}
    logger.info(f"Prompt: {prompt}\nResponse: {response}", extra=extra_data)

def log_ingestion_event(status: str, message: str, count: int = 0):
    """Log an ingestion event to prompts.md"""
    extra_data = {"count": count}
    logger.info(f"Ingestion {status}: {message}", extra=extra_data)