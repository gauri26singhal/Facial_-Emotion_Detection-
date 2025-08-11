
import cv2
import time
from deepface import DeepFace
from gtts import gTTS
import os
import pygame
import threading
import speech_recognition as sr

# Initialize pygame mixer for audio playback
pygame.mixer.init()

# Load face cascade classifier
#face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'qchaarcascade_frontalface_default.xml')
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Start capturing video
cap = cv2.VideoCapture(0)

# Set larger frame width and height
frame_width = 1600
frame_height = 900
cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)

# Initialize variables
face_detected = False
is_playing = False  # Track if audio is currently playing
emotion_text = ""
user_response_text = ""
detection_interval = 5  # Time in seconds between each detection
last_detection_time = time.time()
waiting_for_response = False  # Ensure program waits for user's response

# Initialize the speech recognizer
recognizer = sr.Recognizer()

def listen_for_response():
    """Function to listen for user's voice response and return it as text."""
    with sr.Microphone() as source:
        print("Listening for user response...")
        audio = recognizer.listen(source, timeout=15, phrase_time_limit=15)
        try:
            response = recognizer.recognize_google(audio)
            print("User response:", response)
            return response
        except sr.UnknownValueError:
            print("Could not understand audio")
            return "Sorry, I couldn't understand your response."
        except sr.RequestError:
            print("Error with the speech recognition service")
            return "There was an issue recording your response. Please try again."

def respond_to_user(emotion, user_response):
    """Respond to user based on detected emotion and user's response."""
    global waiting_for_response
    follow_up_messages = {
        'happy': "Ohh that's nice. Nice to see you happy! Keep smiling!",
        'neutral': "Hope you find some excitement today!",
        'sad': "I'm here for you. Remember, every day brings new opportunities.",
        'angry': "Take deep breaths and try to relax. You're stronger than you think.",
        'fear': "Stay strong. Remember, you're safe and supported.",
        'surprise': "I hope your surprise was a pleasant one!",
        'disgust': "I hope things improve soon. Stay positive!"
    }

    # Select the follow-up message based on the detected emotion
    message = follow_up_messages.get(emotion, "Thank you for sharing.")

    # Include user response in the message, or inform them if response couldn't be recorded
    if user_response:
        message += f" You said: {user_response}" if "Sorry" not in user_response else f" {user_response}"

    # Convert the message to speech
    tts = gTTS(text=message, lang='en')
    filename = f"response_{int(time.time())}.mp3"
    tts.save(filename)

    # Play the audio response
    pygame.mixer.music.load(filename)
    pygame.mixer.music.play()

    while pygame.mixer.music.get_busy():
        pygame.time.Clock().tick(10)

    pygame.mixer.music.stop()
    pygame.mixer.music.unload()
    waiting_for_response = False  # Reset to allow the next detection cycle

    if os.path.exists(filename):
        os.remove(filename)

def speak_emotion(emotion):
    """Function to speak the emotion using GTTS."""
    global is_playing, emotion_text, user_response_text, waiting_for_response, last_detection_time
    if is_playing:
        return

    is_playing = True
    waiting_for_response = True
    emotion_text = f"Detected Emotion: {emotion}"
    user_response_text = ""

    emotion_messages = {
        'happy': "You are looking joyful! What brings you joy today?",
        'neutral': "You appear calm. How about bringing some joy into your day?",
        'sad': "I can see that you look a bit sad. Why are you sad?",
        'angry': "You seem a bit upset. What's bringing anger today?",
        'fear': "It looks like you're feeling scared. Why are you scared?",
        'surprise': "Wow! You look surprised! Care to share?",
        'disgust': "You seem a bit disgusted. What’s bothering you?"
    }

    message = emotion_messages.get(emotion, "You have a unique emotion!")

    tts = gTTS(text=message, lang='en')
    filename = f"emotion_{int(time.time())}.mp3"
    tts.save(filename)

    pygame.mixer.music.load(filename)
    pygame.mixer.music.play()

    while pygame.mixer.music.get_busy():
        pygame.time.Clock().tick(10)

    pygame.mixer.music.stop()
    pygame.mixer.music.unload()
    is_playing = False

    if os.path.exists(filename):
        os.remove(filename)

    # Listen for user response
    user_response = listen_for_response()
    user_response_text = f"User Response: {user_response}"
    respond_to_user(emotion, user_response)

    # After response is complete, update the last_detection_time
    last_detection_time = time.time()

# Main loop for camera capture
font_scale = 1.5  # Smaller font scale
font_thickness = 2

while True:
    ret, frame = cap.read()
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    rgb_frame = cv2.cvtColor(gray_frame, cv2.COLOR_GRAY2RGB)
    faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    current_time = time.time()
    time_until_next_detection = int(max(0, detection_interval - (current_time - last_detection_time)))

    # Trigger detection only if no response is pending
    if len(faces) > 0 and time_until_next_detection == 0 and not waiting_for_response:
        x, y, w, h = faces[0]
        face_roi = rgb_frame[y:y + h, x:x + w]

        try:
            result = DeepFace.analyze(face_roi, actions=['emotion'], enforce_detection=False)
            if result:
                detected_emotion = result[0]['dominant_emotion']
                threading.Thread(target=speak_emotion, args=(detected_emotion,)).start()
            else:
                print("No emotion detected")
        except Exception as e:
            print("Error analyzing face:", e)

    # Draw face rectangle and text near the middle of the lower third
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)

    # Set base position higher on the frame
    base_y = int(frame_height - 220)  # Higher Y position for better visibility

    # Display texts with spacing
    cv2.putText(frame, f"Next detection in: {time_until_next_detection}s", (50, base_y),
               cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), font_thickness)
    cv2.putText(frame, emotion_text, (50, base_y - 50), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 0), font_thickness)
    cv2.putText(frame, user_response_text, (50, base_y - 100), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 255), font_thickness)

    # Show video frame
    cv2.imshow('Real-time Emotion Detection', frame)

    # Break loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
pygame.mixer.quit()