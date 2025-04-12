import cv2
import asyncio
import websockets
import mediapipe as mp
import numpy as np


mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False,
                       max_num_hands=1,
                       min_detection_confidence=0.7)


def detect_command_from_landmarks(landmarks):
    if landmarks is None:
        return None

    def is_finger_folded(tip_id, pip_id):
        return landmarks[tip_id].y > landmarks[pip_id].y

    
    finger_tips = [8, 12, 16, 20]   
    finger_pips = [6, 10, 14, 18]

    folded = 0
    for tip, pip in zip(finger_tips, finger_pips):
        if is_finger_folded(tip, pip):
            folded += 1

    print(f"[DEBUG] Folded fingers: {folded}")

    
    is_index_up = not is_finger_folded(8, 6)
    is_middle_up = not is_finger_folded(12, 10)
    is_ring_folded = is_finger_folded(16, 14)
    is_pinky_folded = is_finger_folded(20, 18)

    if is_index_up and is_middle_up and is_ring_folded and is_pinky_folded:
        print("[DEBUG]  Two-finger gesture detected → EXIT")
        return "EXIT"

    
    if folded == 0:
        print("[DEBUG]   FIRE")
        return "FIRE"

    if folded >= 4:
        print("[DEBUG]   ENTER")
        return "ENTER"

   
    wrist = landmarks[0]
    index_tip = landmarks[8]
    dx = index_tip.x - wrist.x
    print(f"[DEBUG] dx = {dx:.3f}")

    if dx < -0.1:
        return "LEFT"
    elif dx > 0.1:
        return "RIGHT"

    return None


async def send_commands():
    uri = "ws://localhost:8765"
    print("Connecting to WebSocket server...")

    async with websockets.connect(uri) as websocket:
        print("Connected to WebSocket server!")

        cap = cv2.VideoCapture(0)

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                continue

            
            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb)

            command = None
            if results.multi_hand_landmarks:
                hand_landmarks = results.multi_hand_landmarks[0]
                command = detect_command_from_landmarks(hand_landmarks.landmark)

                for hand_landmarks in results.multi_hand_landmarks:
                    mp.solutions.drawing_utils.draw_landmarks(
                        frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            
            if command:
                print(f"Detected: {command}")

                
                if command == "EXIT":
                    print("Exit gesture detected. Exiting...")
                    break

                
                try:
                    await websocket.send(command)
                    print(f"Sent: {command}")
                except Exception as e:
                    print(f"[ERROR] WebSocket send failed: {e}")

            
            cv2.imshow("Camera Control", frame)

        

        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    asyncio.run(send_commands())
