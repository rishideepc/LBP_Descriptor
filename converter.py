import os
import cv2
import imageio

def convert_gif_to_jpg(gif_path, output_directory):
    # Create the output directory if it doesn't exist
    
    os.makedirs(output_directory, exist_ok=True)

    # Read the GIF using imageio
    gif_reader = imageio.get_reader(gif_path)

    for frame_index in range(len(gif_reader)):
        # Read a frame
        frame = gif_reader.get_data(frame_index)

        # Convert the frame to RGB (OpenCV uses BGR by default)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Save the frame as a JPG image in the output directory
        output_path = os.path.join(output_directory, f"frame_{frame_index + 1}.jpg")
        cv2.imwrite(output_path, rgb_frame)


directory = 'C:/Users/HP/Desktop/Python_AI/lbp-descriptor-textureRecog/assets/OASIS_MRI_DB'
output_directory = 'C:/Users/HP/Desktop/Python_AI/lbp-descriptor-textureRecog/assets/OASIS_MRI_DB/OASIS_Cross_gallery_converted'
index=0

category_path = os.path.join(directory, f'OASIS_Cross_gallery')
for filename in os.listdir(category_path):
    gif_path = os.path.join(category_path, filename)
    gif_reader = imageio.get_reader(gif_path)
    for frame_index in range(len(gif_reader)):
        # Read a frame
        frame = gif_reader.get_data(frame_index)

        # Convert the frame to RGB (OpenCV uses BGR by default)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Save the frame as a JPG image in the output directory
        output_path = os.path.join(output_directory, f"frame_{index + 1}.jpg")
        cv2.imwrite(output_path, rgb_frame)
    index=index+1
