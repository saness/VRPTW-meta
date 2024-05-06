"""
file: creategif.py
description: This program creates gif from a collection of images
language: python3
author: Sanish Suwal(ss4657@rit.edu), Jay Nair(an1147@rit.edu), Bhavdeep Khileri(bk2281@rit.edu)
"""

import os
import re
import imageio


def natural_sort_key(s):
    """
    Key function for natural sorting.
    """
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', s)]


def create_gif(directory_name, gif_filename):
    """
    Creates gif from the images for best path
    :param directory_name:
    :param gif_filename:
    :return:
    """
    # Get all PNG files in the directory
    png_files = [f for f in os.listdir(directory_name) if f.endswith('.png')]

    # Sort the PNG files by name
    png_files.sort(key=natural_sort_key)

    images = []
    for png_file in png_files:
        # print(png_file)
        file_path = os.path.join(directory_name, png_file)
        print(file_path)
        images.append(imageio.imread(file_path))
    # Save the GIF
    imageio.mimsave(gif_filename, images, fps=5)  # Duration is in seconds, here set to 1 second per frame


# Example usage
if __name__ == "__main__":
    directory_name = "assets/image"  # Change to the directory containing your PNG images
    gif_filename = "output.gif"

    create_gif(directory_name, gif_filename)