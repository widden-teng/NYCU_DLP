
import os
from PIL import Image

def concat_gifs_horizontally(gif_paths, output_path):
    gif_images = [Image.open(gif_path) for gif_path in gif_paths]
    frames = []
    
    while True:
        try:
            current_frames = [img.resize((img.width, img.height), Image.ANTIALIAS) for img in gif_images]
            max_height = max(img.height for img in current_frames)

            total_width = sum(img.width for img in current_frames)
            combined_image = Image.new("RGBA", (total_width, max_height))

            x_offset = 0
            for img in current_frames:
                combined_image.paste(img, (x_offset, 0))
                x_offset += img.width

            frames.append(combined_image)

            for img in gif_images:
                img.seek(img.tell() + 1)
        except EOFError:
            break

    frames[0].save(output_path, save_all=True, append_images=frames[1:], optimize=False, duration=300, loop=0)


for i in range(6):
    image_folder = f'./img/{i}/'
    image_files = sorted([f for f in os.listdir(image_folder) if f.endswith('.png') or f.endswith('.jpg')])
    images = [Image.open(os.path.join(image_folder, image_file)) for image_file in image_files]

    gif_path = f'./gif{i+1}.gif'
    images[0].save(gif_path, save_all=True, append_images=images[1:], optimize=False, duration=300, loop=0)

gif_paths = ["gif1.gif", "gif2.gif", "gif3.gif", "gif4.gif", "gif5.gif", "gif6.gif"]
output_path = "combined_gif.gif"
concat_gifs_horizontally(gif_paths, output_path)


for fold in ["./img/GT", "./img/Pred"]:
    folder_path = fold
    image_files = os.listdir(folder_path)

    images = [Image.open(os.path.join(folder_path, img_file)) for img_file in image_files[:12]]

    total_width = sum([img.width for img in images])
    max_height = max([img.height for img in images])

    concatenated_image = Image.new('RGB', (total_width, max_height))

    x_offset = 0
    for img in images:
        concatenated_image.paste(img, (x_offset, 0))
        x_offset += img.width

    concatenated_image.save(f'{fold[6:]}_image.jpg')


