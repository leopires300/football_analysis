from utils import read_video, save_video
from trackers import Tracker
import pandas as pd
import numpy as np

def main():
    # Read Video
    video_frames = read_video('input_videos/08fd33_4.mp4')

    #initialize tracker
    tracker = Tracker('models/best.pt')
    tracks = tracker.get_object_tracks(video_frames,
                                       read_from_stub=True,
                                        stub_path='stubs/tracks_stubs.pkl')
    
    tmp = pd.DataFrame.from_dict(tracks)
    print(f"Total de entradas em tracks: {np.shape(tmp)}")

    # Draw output
    ## Draw object tracks
    output_video_frames = tracker.draw_annotations(video_frames, tracks)
    
    # Save Video
    save_video(output_video_frames, 'output_video/ouput_video.avi')

if __name__ == '__main__':
    main()