from utils import read_video, save_video
from trackers import Tracker
import pandas as pd
import numpy as np
import cv2
from team_assigner import TeamAssigner
from player_ball_assigner import PlayerBallAssigner

def main():
    # Read Video
    video_frames = read_video('input_videos/08fd33_41.mp4')

    #initialize tracker
    tracker = Tracker('models/best.pt')
    tracks = tracker.get_object_tracks(video_frames,
                                       read_from_stub=True,
                                        stub_path='stubs/tracks_stubs.pkl')

    tmp = pd.DataFrame.from_dict(tracks)
    print(f"Total de entradas em tracks: {np.shape(tmp)}")

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
    team_ball_control = []
    for frame_num, player_track in enumerate(tracks['players']):
        ball_bbox = tracks['ball'][frame_num][1]['bbox']
        assigned_player = players_assigner.assign_ball_to_player(player_track, ball_bbox)

        if assigned_player != -1:
            # tracks['ball'][frame_num][1]['assigned_player'] = assigned_player
            tracks['players'][frame_num][assigned_player]['has_ball'] = True
            team_ball_control.append(tracks['players'][frame_num][assigned_player]['team'])
        else:
            team_ball_control.append(team_ball_control[-1])
    
    team_ball_control = np.array(team_ball_control)

    # Draw output
    ## Draw object tracks
    output_video_frames = tracker.draw_annotations(video_frames, tracks, team_ball_control)

    # Save Video
    save_video(output_video_frames, 'output_video/Interpolation_08fd33_41.avi')

if __name__ == '__main__':
    main()