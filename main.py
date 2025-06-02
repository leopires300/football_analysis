from utils import read_video, save_video
from trackers import Tracker
import pandas as pd
import numpy as np
import cv2
from team_assigner import TeamAssigner
from player_ball_assigner import PlayerBallAssigner
from camera_movement_estimator import CameraMovementEstimator

def main():
    # Read Video
    video_frames = read_video('input_videos/08fd33_41.mp4')

    #initialize tracker
    tracker = Tracker('models/best.pt')
    tracks = tracker.get_object_tracks(video_frames,
                                       read_from_stub=True,
                                        stub_path='stubs/tracks_stubs.pkl')

    # Get object positions
    tracker.add_position_to_tracks(tracks)

    tmp = pd.DataFrame.from_dict(tracks)
    print(f"Total de entradas em tracks: {np.shape(tmp)}")

    # Get camera movement
    camera_movement_estimator = CameraMovementEstimator(video_frames[0])
    camera_movement_per_frame = camera_movement_estimator.get_camera_movement(video_frames,
                                                                                    read_from_stub=True,
                                                                                    stub_path='stubs/camera_movement_stub.pkl')

    # Add adjusted positions to tracks
    camera_movement_estimator.add_adjust_positions_to_tracks(tracks, camera_movement_per_frame)
    print(f"Total de entradas em camera_movement_per_frame: {len(camera_movement_per_frame)}")

    # Interpolate ball positions
    tracks['ball'] = tracker.interpolate_ball_positions(tracks['ball'])

    team_assigner = TeamAssigner()
    team_assigner.assign_team_color(video_frames[0], tracks['players'][0])

    for frame_num, player_track in enumerate(tracks['players']):
        for player_id, track in player_track.items():
            team = team_assigner.get_player_team(video_frames[frame_num],
                                                 track['bbox'],
                                                 player_id)
            tracks['players'][frame_num][player_id]['team'] = team
            tracks['players'][frame_num][player_id]['team_color'] = team_assigner.team_colors[team]
    
    #Assign ball to Aquisition
    players_assigner = PlayerBallAssigner()
    team_ball_control = [] #-1
    for frame_num, player_track in enumerate(tracks['players']):
        ball_bbox = tracks['ball'][frame_num][1]['bbox']
        assigned_player = players_assigner.assign_ball_to_player(player_track, ball_bbox)

        if assigned_player != -1:
            # tracks['ball'][frame_num][1]['assigned_player'] = assigned_player
            tracks['players'][frame_num][assigned_player]['has_ball'] = True
            team_ball_control.append(tracks['players'][frame_num][assigned_player]['team'])
        else:
            if len(team_ball_control) == 0:
                team_ball_control.append(-1)
            else:
                team_ball_control.append(team_ball_control[-1])
    
    team_ball_control = np.array(team_ball_control)

    # Draw output
    ## Draw object tracks
    output_video_frames = tracker.draw_annotations(video_frames, tracks, team_ball_control)

    ## Draw camera movement
    output_video_frames = camera_movement_estimator.draw_camera_movement(output_video_frames, camera_movement_per_frame)

    # Save Video
    save_video(output_video_frames, 'output_video/123.avi')

if __name__ == '__main__':
    main()