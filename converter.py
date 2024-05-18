import os
import cv2
import imageio

def convert_gif_to_jpg(gif_path, output_directory):
    
    os.makedirs(output_directory, exist_ok=True)
    gif_reader = imageio.get_reader(gif_path)

    for frame_index in range(len(gif_reader)):
        frame = gif_reader.get_data(frame_index)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
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
        frame = gif_reader.get_data(frame_index)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        output_path = os.path.join(output_directory, f"frame_{index + 1}.jpg")
        cv2.imwrite(output_path, rgb_frame)
    index=index+1
