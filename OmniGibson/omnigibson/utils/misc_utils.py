import os
import cv2
import numpy as np

from moviepy import VideoFileClip, concatenate_videoclips


def hori_concatenate_image(images):
    # Ensure the images have the same height
    image1 = images[0]
    concatenated_image = image1
    for i in range(1, len(images)):
        image_i = images[i]
        if image1.shape[0] != image_i.shape[0]:
            # print("Images do not have the same height. Resizing the second image.")
            height = image1.shape[0]
            image_i = cv2.resize(image_i, (int(image_i.shape[1] * (height / image_i.shape[0])), height))

        # Concatenate the images side by side
        concatenated_image = np.concatenate((concatenated_image, image_i), axis=1)

    return np.array(concatenated_image)


def combine_videos(folder_path, output_path):
    # Get all MP4 files from the folder and sort them
    video_files = sorted([os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith(".mp4")])

    # Load video clips
    clips = [VideoFileClip(video) for video in video_files]

    # Concatenate videos
    final_video = concatenate_videoclips(clips)

    # Save the merged video
    final_video.write_videofile(f"{output_path}/merged_video.mp4", codec="libx264", fps=24)

    # Close clips
    for clip in clips:
        clip.close()