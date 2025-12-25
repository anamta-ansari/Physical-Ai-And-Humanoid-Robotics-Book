---
title: Voice-to-Action
sidebar_position: 2
description: Converting speech commands to robotic actions using OpenAI Whisper and natural language understanding
---

# Voice-to-Action

## Using OpenAI Whisper

OpenAI Whisper is a state-of-the-art automatic speech recognition (ASR) system that converts spoken language into text. For robotics applications, Whisper serves as a crucial component in the voice-to-action pipeline by providing accurate speech-to-text conversion.

### Whisper Architecture

Whisper is built on a Transformer-based architecture that jointly learns speech and text representations:

- **Encoder**: Processes audio input using a convolutional neural network followed by Transformer layers
- **Decoder**: Generates text output conditioned on the encoded audio
- **Multilingual Capability**: Trained on multiple languages simultaneously
- **Robustness**: Performs well on low-quality audio and accented speech

### Key Features for Robotics

#### High Accuracy
- State-of-the-art performance across multiple languages
- Robust to background noise and audio quality variations
- Handles various accents and speaking styles

#### Real-time Processing
- Optimized for low-latency inference
- Suitable for interactive robotic applications
- Can run on edge devices with appropriate optimization

#### Multilingual Support
- Supports over 98 languages
- Can identify the language being spoken
- Useful for diverse user populations

### Implementing Whisper in Robotics

Here's an example of integrating Whisper into a robotic voice-to-action system:

```python
import whisper
import torch
import rospy
from std_msgs.msg import String

class VoiceToActionNode:
    def __init__(self):
        rospy.init_node('voice_to_action')
        
        # Load Whisper model (choose appropriate size for your hardware)
        self.model = whisper.load_model("base.en")  # For English-only
        # self.model = whisper.load_model("base")   # For multilingual
        
        # Publishers and subscribers
        self.audio_sub = rospy.Subscriber('/audio_input', AudioData, self.audio_callback)
        self.command_pub = rospy.Publisher('/robot_commands', String, queue_size=10)
        
        # Robot state and capabilities
        self.robot_capabilities = {
            "move_forward": self.move_forward,
            "turn_left": self.turn_left,
            "turn_right": self.turn_right,
            "pick_up_object": self.pick_up_object,
            "place_object": self.place_object,
            "stop": self.stop_robot
        }
    
    def audio_callback(self, audio_data):
        # Convert audio data to numpy array
        audio_array = self.convert_audio_to_array(audio_data)
        
        # Transcribe using Whisper
        result = self.model.transcribe(audio_array)
        text = result["text"]
        
        # Process the transcribed text to extract commands
        self.process_command(text)
    
    def process_command(self, text):
        # Simple keyword matching (can be enhanced with NLP)
        text_lower = text.lower().strip()
        
        if "move forward" in text_lower or "go forward" in text_lower:
            self.command_pub.publish("move_forward")
        elif "turn left" in text_lower:
            self.command_pub.publish("turn_left")
        elif "turn right" in text_lower:
            self.command_pub.publish("turn_right")
        elif "pick up" in text_lower or "grasp" in text_lower:
            self.command_pub.publish("pick_up_object")
        elif "place" in text_lower or "put down" in text_lower:
            self.command_pub.publish("place_object")
        elif "stop" in text_lower or "halt" in text_lower:
            self.command_pub.publish("stop")
        else:
            rospy.loginfo(f"Unrecognized command: {text}")
    
    def convert_audio_to_array(self, audio_data):
        # Convert ROS audio message to numpy array for Whisper
        # Implementation depends on your audio format
        import numpy as np
        # This is a simplified example - actual implementation depends on audio format
        audio_array = np.frombuffer(audio_data.data, dtype=np.int16).astype(np.float32)
        audio_array /= 32768.0  # Normalize to [-1, 1]
        return audio_array
    
    def move_forward(self):
        # Implementation for moving robot forward
        pass
    
    def turn_left(self):
        # Implementation for turning robot left
        pass
    
    def turn_right(self):
        # Implementation for turning robot right
        pass
    
    def pick_up_object(self):
        # Implementation for picking up an object
        pass
    
    def place_object(self):
        # Implementation for placing an object
        pass
    
    def stop_robot(self):
        # Implementation for stopping robot
        pass

if __name__ == '__main__':
    node = VoiceToActionNode()
    rospy.spin()
```

### Whisper Model Variants

Whisper comes in different sizes optimized for different use cases:

| Model | Size | Required VRAM | Relative Speed | English-only | Multilingual |
|-------|------|---------------|----------------|--------------|--------------|
| tiny  | 75 MB | ~1 GB | 32x | ✓ | ✓ |
| base  | 145 MB | ~1 GB | 16x | ✓ | ✓ |
| small | 465 MB | ~2 GB | 6x | ✓ | ✓ |
| medium | 1.5 GB | ~5 GB | 2x | ✓ | ✓ |
| large | 3.0 GB | ~10 GB | 1x | ✗ | ✓ |

For robotics applications, the choice depends on:
- Available computational resources
- Required accuracy
- Real-time constraints
- Language requirements

## Converting speech to commands

Converting speech to robotic commands involves several processing steps beyond basic speech recognition:

### Speech Recognition Pipeline

```mermaid
graph LR
    A[Raw Audio] --> B[Preprocessing]
    B --> C[ASR (Whisper)]
    C --> D[Text]
    D --> E[NLP Processing]
    E --> F[Commands]
    F --> G[Robot Actions]

    style A fill:#e1f5fe
    style G fill:#e8f5e8
```

### Natural Language Processing for Commands

Once speech is converted to text, it needs to be processed to extract actionable commands:

```python
import re
from typing import List, Dict, Tuple

class CommandExtractor:
    def __init__(self):
        # Define command patterns
        self.command_patterns = {
            "navigation": [
                (r"go to (.+)", "navigate_to"),
                (r"move to (.+)", "navigate_to"),
                (r"go (.+)", "move_direction"),
                (r"move (.+)", "move_direction"),
            ],
            "manipulation": [
                (r"pick up (.+)", "pick_up"),
                (r"grasp (.+)", "pick_up"),
                (r"take (.+)", "pick_up"),
                (r"place (.+)", "place"),
                (r"put (.+)", "place"),
            ],
            "interaction": [
                (r"say (.+)", "speak"),
                (r"tell (.+)", "speak"),
                (r"introduce yourself", "introduce"),
            ]
        }
    
    def extract_commands(self, text: str) -> List[Dict]:
        commands = []
        text_lower = text.lower()
        
        for category, patterns in self.command_patterns.items():
            for pattern, command_type in patterns:
                matches = re.findall(pattern, text_lower)
                for match in matches:
                    commands.append({
                        "type": command_type,
                        "target": match if isinstance(match, str) else match[0] if isinstance(match, tuple) else "",
                        "original_text": text
                    })
        
        return commands

# Example usage
extractor = CommandExtractor()
commands = extractor.extract_commands("Please go to the kitchen and pick up the red cup")
print(commands)
# Output: [{'type': 'navigate_to', 'target': 'the kitchen', 'original_text': 'Please go to the kitchen and pick up the red cup'}, 
#         {'type': 'pick_up', 'target': 'the red cup', 'original_text': 'Please go to the kitchen and pick up the red cup'}]
```

### Context-Aware Command Processing

Robotic commands often need to be interpreted in context:

```python
class ContextAwareProcessor:
    def __init__(self):
        self.current_location = "unknown"
        self.robot_capabilities = ["navigation", "manipulation", "speech"]
        self.known_objects = ["red cup", "blue box", "green bottle"]
        self.known_locations = ["kitchen", "living room", "bedroom", "office"]
    
    def process_command_with_context(self, command, context):
        # Validate command against robot capabilities
        if command["type"] not in self.robot_capabilities:
            return {"error": f"Robot cannot perform {command['type']}", "action": None}
        
        # Resolve ambiguous references
        if command["target"] in ["it", "that", "this"]:
            command["target"] = context.get("last_mentioned_object", "unknown")
        
        # Validate target location
        if command["type"] == "navigate_to" and command["target"] not in self.known_locations:
            return {"error": f"Unknown location: {command['target']}", "action": None}
        
        # Validate target object
        if command["type"] in ["pick_up", "place"] and command["target"] not in self.known_objects:
            return {"error": f"Unknown object: {command['target']}", "action": None}
        
        # Generate action plan
        return {
            "action": self.generate_action_plan(command),
            "success": True
        }
    
    def generate_action_plan(self, command):
        # Generate a sequence of actions to fulfill the command
        if command["type"] == "navigate_to":
            return {
                "action_sequence": [
                    {"type": "path_planning", "target": command["target"]},
                    {"type": "navigation", "target": command["target"]}
                ]
            }
        elif command["type"] == "pick_up":
            return {
                "action_sequence": [
                    {"type": "object_detection", "target": command["target"]},
                    {"type": "approach_object", "target": command["target"]},
                    {"type": "grasp_object", "target": command["target"]}
                ]
            }
        # Add more action types as needed
```

## Natural language understanding

Natural Language Understanding (NLU) is crucial for interpreting human commands in a robotic context. It goes beyond simple keyword matching to understand intent, entities, and context.

### Intent Recognition

Identifying the user's intent from spoken commands:

```python
from transformers import pipeline

class IntentRecognizer:
    def __init__(self):
        # Using a pre-trained model for intent classification
        self.classifier = pipeline(
            "text-classification",
            model="microsoft/DialoGPT-medium",  # This is an example - you'd use a task-specific model
            return_all_scores=True
        )
        
        # Define possible intents
        self.intents = {
            "navigation": ["go to", "move to", "navigate to", "travel to"],
            "manipulation": ["pick up", "grasp", "take", "place", "put", "move"],
            "information": ["what is", "tell me about", "describe", "show me"],
            "social": ["hello", "goodbye", "introduce", "name"],
            "help": ["help", "what can you do", "how to"]
        }
    
    def recognize_intent(self, text):
        # Simple keyword-based intent recognition
        text_lower = text.lower()
        
        for intent, keywords in self.intents.items():
            for keyword in keywords:
                if keyword in text_lower:
                    return intent
        
        return "unknown"
```

### Entity Recognition

Identifying specific objects, locations, or parameters in commands:

```python
import spacy

class EntityRecognizer:
    def __init__(self):
        # Load spaCy model for English
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            print("Please install spaCy English model: python -m spacy download en_core_web_sm")
            self.nlp = None
    
    def extract_entities(self, text):
        if not self.nlp:
            return {"objects": [], "locations": [], "quantities": []}
        
        doc = self.nlp(text)
        
        entities = {
            "objects": [],
            "locations": [],
            "quantities": [],
            "people": []
        }
        
        for ent in doc.ents:
            if ent.label_ in ["OBJECT", "PRODUCT"]:  # Custom labels would need to be trained
                entities["objects"].append(ent.text)
            elif ent.label_ in ["GPE", "LOC", "FAC"]:  # Geographic, location, facility
                entities["locations"].append(ent.text)
            elif ent.label_ in ["MONEY", "QUANTITY", "CARDINAL", "ORDINAL"]:
                entities["quantities"].append(ent.text)
            elif ent.label_ in ["PERSON"]:
                entities["people"].append(ent.text)
        
        # Also extract using patterns
        import re
        # Extract color + object patterns
        color_object_pattern = r"(red|blue|green|yellow|black|white|large|small|big|little)\s+(\w+)"
        color_objects = re.findall(color_object_pattern, text.lower())
        for color, obj in color_objects:
            entities["objects"].append(f"{color} {obj}")
        
        return entities
```

### Dialogue Management

Managing multi-turn conversations for complex commands:

```python
class DialogueManager:
    def __init__(self):
        self.conversation_state = {}
        self.pending_requests = {}
    
    def process_input(self, user_input, user_id):
        if user_id not in self.conversation_state:
            self.conversation_state[user_id] = {"context": {}, "history": []}
        
        state = self.conversation_state[user_id]
        
        # Check if this continues a previous request
        if user_id in self.pending_requests:
            return self.handle_continuation(user_input, user_id)
        
        # Process as new request
        intent = self.recognize_intent(user_input)
        
        if intent == "navigation":
            return self.handle_navigation_request(user_input, state)
        elif intent == "manipulation":
            return self.handle_manipulation_request(user_input, state)
        else:
            return {"response": "I can help with navigation and manipulation tasks. What would you like me to do?"}
    
    def handle_navigation_request(self, user_input, state):
        # Extract target location
        entities = self.extract_entities(user_input)
        
        if not entities["locations"]:
            # Ask for clarification
            self.pending_requests[state["user_id"]] = {"type": "navigation", "partial": user_input}
            return {"request": "Where would you like me to go?"}
        
        target_location = entities["locations"][0]
        return {"action": "navigate", "target": target_location}
    
    def handle_continuation(self, user_input, user_id):
        pending = self.pending_requests[user_id]
        
        if pending["type"] == "navigation":
            # Complete the navigation request
            del self.pending_requests[user_id]
            return {"action": "navigate", "target": user_input}
```

## Safety in voice control

Safety is paramount in voice-controlled robotics, especially when robots operate in human environments.

### Command Validation

Validating commands before execution:

```python
class SafetyValidator:
    def __init__(self):
        self.forbidden_commands = [
            "harm", "injure", "break", "destroy", "attack", "hit", "hurt"
        ]
        self.safe_zones = ["designated_area", "workspace", "lab"]  # Define safe operational areas
        self.emergency_stop_keywords = ["emergency", "stop", "halt", "help"]
    
    def validate_command(self, command, context):
        # Check for forbidden language
        text_lower = command.get("original_text", "").lower()
        for forbidden in self.forbidden_commands:
            if forbidden in text_lower:
                return {
                    "valid": False, 
                    "reason": "Command contains potentially harmful language",
                    "safety_action": "ignore"
                }
        
        # Check if robot is in safe operating area
        if command["type"] in ["navigation", "manipulation"]:
            if context.get("location") not in self.safe_zones:
                return {
                    "valid": False,
                    "reason": "Robot is not in a safe area for this operation",
                    "safety_action": "request_permission"
                }
        
        # Check for emergency stop
        for keyword in self.emergency_stop_keywords:
            if keyword in text_lower:
                return {
                    "valid": True,
                    "action": "emergency_stop",
                    "priority": "high"
                }
        
        return {"valid": True, "action": command["type"]}
```

### Safety Mechanisms

Implementing safety measures in the voice-to-action pipeline:

```python
class SafeVoiceToAction:
    def __init__(self):
        self.safety_validator = SafetyValidator()
        self.command_extractor = CommandExtractor()
        self.context_processor = ContextAwareProcessor()
        
    def process_voice_command(self, audio_input):
        # Step 1: Convert speech to text using Whisper
        text = self.speech_to_text(audio_input)
        
        # Step 2: Extract commands
        commands = self.command_extractor.extract_commands(text)
        
        # Step 3: Validate for safety
        validated_commands = []
        for cmd in commands:
            validation_result = self.safety_validator.validate_command(cmd, self.get_robot_context())
            
            if validation_result["valid"]:
                if validation_result.get("action") == "emergency_stop":
                    self.execute_emergency_stop()
                    return {"status": "emergency_stop_executed"}
                else:
                    validated_commands.append(cmd)
            else:
                rospy.logwarn(f"Unsafe command blocked: {validation_result['reason']}")
        
        # Step 4: Process validated commands
        results = []
        for cmd in validated_commands:
            result = self.context_processor.process_command_with_context(cmd, self.get_robot_context())
            if result["success"]:
                self.execute_command(result["action"])
                results.append({"status": "success", "command": cmd})
            else:
                results.append({"status": "failed", "error": result.get("error")})
        
        return results
    
    def speech_to_text(self, audio_input):
        # Implementation using Whisper
        pass
    
    def execute_command(self, action_plan):
        # Execute the action plan on the robot
        pass
    
    def execute_emergency_stop(self):
        # Immediately stop all robot motion
        pass
    
    def get_robot_context(self):
        # Return current robot state, location, etc.
        return {
            "location": "unknown",  # Would come from localization system
            "battery_level": 100,   # Would come from robot status
            "current_task": "idle"  # Would come from task manager
        }
```

## Integration with Robot Systems

Integrating voice-to-action systems with existing robot architectures:

### ROS Integration Example

```xml
<!-- launch file for voice-to-action node -->
<launch>
  <!-- Audio input node -->
  <node name="audio_input" pkg="audio_capture" type="audio_capture_node" />
  
  <!-- Voice-to-action node -->
  <node name="voice_to_action" pkg="voice_control" type="voice_to_action_node.py" output="screen">
    <param name="whisper_model" value="base.en" />
    <param name="robot_name" value="my_robot" />
  </node>
  
  <!-- Navigation stack -->
  <include file="$(find my_robot_navigation)/launch/navigation.launch" />
  
  <!-- Manipulation stack -->
  <include file="$(find my_robot_manipulation)/launch/manipulation.launch" />
</launch>
```

### Error Handling and Feedback

Providing appropriate feedback to users:

```python
class VoiceControlFeedback:
    def __init__(self):
        self.speech_publisher = rospy.Publisher('/speech_output', String, queue_size=10)
        self.led_publisher = rospy.Publisher('/status_led', ColorRGBA, queue_size=10)
    
    def provide_feedback(self, result):
        if result["status"] == "success":
            self.speak("I have completed the requested action")
            self.set_led_color(green=True)
        elif result["status"] == "failed":
            error_msg = result.get("error", "The requested action could not be completed")
            self.speak(f"Sorry, I couldn't do that: {error_msg}")
            self.set_led_color(red=True)
        elif result["status"] == "pending":
            self.speak("I'm working on your request")
            self.set_led_color(yellow=True)
    
    def speak(self, text):
        # Convert text to speech and output audio
        self.speech_publisher.publish(text)
    
    def set_led_color(self, red=False, green=False, blue=False, yellow=False):
        # Set status LED color for visual feedback
        color_msg = ColorRGBA()
        if yellow:
            color_msg.r = 1.0
            color_msg.g = 1.0
        elif green:
            color_msg.g = 1.0
        elif red:
            color_msg.r = 1.0
        self.led_publisher.publish(color_msg)
```

## Conclusion

Voice-to-action systems represent a significant advancement in human-robot interaction, enabling more natural and intuitive communication. By leveraging technologies like OpenAI Whisper for speech recognition and sophisticated natural language processing, robots can understand and execute complex commands in real-world environments.

The implementation of such systems requires careful consideration of safety, validation, and error handling to ensure reliable operation. As these technologies continue to advance, we can expect voice-controlled robots to become increasingly sophisticated and capable of handling complex tasks in diverse environments.

The integration of voice control with other robotic systems like navigation and manipulation creates powerful platforms for assistive robotics, industrial automation, and many other applications where natural human-robot interaction is valuable.