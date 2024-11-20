import cv2
import mediapipe as mp
from model_architectures.autoencoder import SingleAutoencoder

if __name__ == '__main__':

    autoencoder = SingleAutoencoder(input_dimension=224,
                                    summarize_model=True,
                                    patch_size=4,
                                    expansion_factor=2,
                                    pre_trained_path=None)

    '''
    # Initialize MediaPipe Face Mesh
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, min_detection_confidence=0.5)

    # Initialize MediaPipe drawing utilities
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles

    # Start capturing from webcam
    cap = cv2.VideoCapture(0)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame.")
            break

        # Convert the frame to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process the frame to detect face landmarks
        results = face_mesh.process(rgb_frame)

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                # Draw landmarks on the frame (optional)
                mp_drawing.draw_landmarks(
                    frame,
                    face_landmarks,
                    mp_face_mesh.FACEMESH_TESSELATION,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style(),
                )

                # Print the coordinates of landmarks
                for idx, landmark in enumerate(face_landmarks.landmark):
                    x = int(landmark.x * frame.shape[1])  # Convert to pixel coordinates
                    y = int(landmark.y * frame.shape[0])
                    z = landmark.z  # Relative depth (not pixel)
                    print(f"Landmark {idx}: ({x}, {y}, {z})")

        # Display the frame
        cv2.imshow("Face Mesh", frame)

        # Exit on pressing 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    '''