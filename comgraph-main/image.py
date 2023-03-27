import sys
from PIL import Image
import subprocess
from typing import List
import os
import imgbbpy


def concatanate_images(
    image_path_list: List[str],
    filename_prefix: str = "img/image",
    num_x=3,
    num_y=3,
    print_filename=True,
):
    all_images = [Image.open(x) for x in image_path_list]
    num_images = len(all_images)
    widths, heights = zip(*(i.size for i in all_images))

    NUM_IN_ONE_PAGE = num_x * num_y
    WIDTH = widths[0] // num_x
    HEIGHT = heights[0] // num_x
    NUM_PAGES = (num_images - 1) // NUM_IN_ONE_PAGE + 1
    for page in range(NUM_PAGES):
        images = all_images[page *
                            NUM_IN_ONE_PAGE:(page + 1) * NUM_IN_ONE_PAGE]
        if len(images) >= num_x:
            image_width = WIDTH * num_x
        else:
            image_width = WIDTH * len(images)
        image_height = HEIGHT * min((len(images) - 1) // num_x + 1, num_y)
        new_im = Image.new(
            'RGBA',
            (image_width, image_height),
            (0, 0, 0, 0)  # transparent
        )

        x_offset = 0
        for idx, im in enumerate(images):
            x = (idx % num_x) * WIDTH
            y = (idx // num_x) * HEIGHT
            im = im.resize((WIDTH, HEIGHT))
            new_im.paste(im, (x, y))
            x_offset += im.size[0]
        filename = filename_prefix + '-' + str(page) + '.png'
        new_im.save(filename)
        if print_filename:
            print(filename)
        return filename
    try:
        subprocess.run(["open", filename])
    except FileNotFoundError:
        pass


def convert_heic_to_png(path: str, old_extension='heic', new_extension='.jpeg') -> str:
    new_name = path.replace(old_extension, new_extension)
    heif_file = pyheif.read(path)
    data = Image.frombytes(
        heif_file.mode,
        heif_file.size,
        heif_file.data,
        "raw",
        heif_file.mode,
        heif_file.stride,
    )
    data.save(new_name, new_extension[1:])
    return new_name


def upload_to_imgbb(path_to_file: str):
    _, extension = os.path.splitext(path_to_file)
    if extension == ".heic" or extension == ".HEIC":
        path_to_file = convert_heic_to_png(path_to_file, extension)
    # Need to set `export IMGBB_API_KEY=xxxxyyyyyzzzzz` in ~/.zshrc
    IMGBB_API_KEY = os.environ["IMGBB_API_KEY"]
    client = imgbbpy.SyncClient(IMGBB_API_KEY)
    image = client.upload(file=path_to_file)
    print(image.url)
    return image.url


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("usage: python3", sys.argv[0],
              "<path to image>")
        exit(0)
    for filename in sys.argv[1:]:
        upload_to_imgbb(filename)
