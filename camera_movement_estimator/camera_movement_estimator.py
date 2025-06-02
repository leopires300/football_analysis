import pickle
import cv2
import numpy as np
import os
import sys
sys.path.append('../')
from utils import measure_distance, measure_xy_distance

class CameraMovementEstimator():
    def __init__(self,frame):
        
        self.minimum_distance = 5

        self.lk_params = dict(
            winSize=(15, 15),
            maxLevel=2,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03),
        )

        first_frame_grayscale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        height, width = first_frame_grayscale.shape
        mask = np.zeros_like(first_frame_grayscale)

        margin = int(height * 0.1)
        mask[:, 0:margin] = 1
        mask[:, (height - margin):height] = 1

        self.features = dict(
            maxCorners=200,
            qualityLevel=0.01,
            minDistance=2,
            blockSize=3,
            mask=mask,
        )
    def add_adjust_positions_to_tracks(self, tracks, camera_movement_per_frame):
        for object, object_tracks in tracks.items():
            for frame_num, track in enumerate(object_tracks):
                for track_id, track_info in track.items():
                    position = track_info['position']
                    camera_movement = camera_movement_per_frame[frame_num]
                    adjusted_position = [position[0] - camera_movement[0], position[1] - camera_movement[1]]
                    tracks[object][frame_num][track_id]['adjusted_position'] = adjusted_position
    
    def get_camera_movement(self, frames, read_from_stub = False, stub_path = None):
        # Read the stub
        if read_from_stub and stub_path is not None and os.path.exists(stub_path):
            with open(stub_path, 'rb') as f:
                camera_movement = pickle.load(f)
                return camera_movement

        # Initialize the camera movement list
        camera_movement = [[0,0]]*len(frames)

        old_gray = cv2.cvtColor(frames[0], cv2.COLOR_BGR2GRAY)
        old_features = cv2.goodFeaturesToTrack(old_gray, **self.features)

        for frame_num in range(1, len(frames)):

            new_gray = cv2.cvtColor(frames[frame_num], cv2.COLOR_BGR2GRAY)
            new_features, _, _ = cv2.calcOpticalFlowPyrLK(old_gray, new_gray, old_features, None, **self.lk_params)

            max_distance = 0
            camera_movement_x, camera_movement_y = 0, 0

            # Calculate the movement
            for i, (new, old) in enumerate(zip(new_features, old_features)):
                new_features_point = new.ravel()
                old_features_point = old.ravel()

                distance = measure_distance(new_features_point, old_features_point)
                if distance > max_distance:
                    max_distance = distance
                    camera_movement_x, camera_movement_y = measure_xy_distance(old_features_point, new_features_point)
                    # camera_movement_y = new_features_point[1] - old_features_point[1]
                    # camera_movement_x = new_features_point[0] - old_features_point[0]

            # Check if the movement is significant
            if max_distance > self.minimum_distance:
                camera_movement[frame_num] = [camera_movement_x, camera_movement_y]
                old_features = cv2.goodFeaturesToTrack(new_gray, **self.features)

            # Update the previous frame and previous points
            old_gray = new_gray.copy()

        if stub_path is not None:
            with open(stub_path, 'wb') as f:
                pickle.dump(camera_movement, f)
    
        return camera_movement
    
    def draw_camera_movement(self, frames, camera_movement_per_frame):
        output_frames = []
        
        for frame_num, frame in enumerate(frames):
            overlay = frame.copy()
            cv2.rectangle(overlay, (30,30), (510,100), (0,0,0), -1)
            alpha = 0.6
            cv2.addWeighted(overlay, alpha, frame, (1 - alpha), 0, frame)

            x_movement, y_movement = camera_movement_per_frame[frame_num]
            frame = cv2.putText(frame, f"Camera Movement X: {x_movement:.2f}", (40, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            frame = cv2.putText(frame, f"Camera Movement Y: {y_movement:.2f}", (40, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            output_frames.append(frame)
        
        return output_frames