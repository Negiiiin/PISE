from PIL import Image
import os

def crop_center_square(image_path, output_path):
    """
    Crop the center 256x256 square of an image and save it to output_path.
    """
    with Image.open(image_path) as img:
        # Assuming the image is 256 pixels in height and 768 pixels in width
        width, height = img.size

        # Calculate the left, upper, right, and lower pixel coordinate for the middle square
        left = (width - 256) / 2
        upper = (height - 256) / 2
        right = (width + 256) / 2
        lower = (height + 256) / 2

        # Crop the center square
        cropped_img = img.crop((left, upper, right, lower))

        # Save the cropped image
        cropped_img.save(output_path)

# Example usage:
# # Define your input and output directories
# input_directory = "./fashion_data/eval_results"
# output_directory = "./output_images"
#
# if not os.path.exists(output_directory):
#     os.makedirs(output_directory)
#
# # Iterate over all images in the input directory
# for filename in os.listdir(input_directory):
#     if filename.endswith(".jpg"):  # Assuming the images are in JPG format; adjust if necessary
#         input_path = os.path.join(input_directory, filename)
#         output_path = os.path.join(output_directory, f"cropped_{filename}")
#         crop_center_square(input_path, output_path)


from PIL import Image
import os
import math

import os


def find_images_with_word(directory, word):
    """
    Find all images in a directory where the filename contains the specified word.
    The images are sorted by the 'num1' part of the filename, which follows the pattern 'cropped_num1/num2_'.
    """
    matching_files = []
    for filename in os.listdir(directory):
        if word.lower() in filename.lower() and filename.endswith((".jpg", ".png", ".jpeg")):
            matching_files.append(filename)

    # Function to extract 'num1' from the filename
    def extract_num1(filename):
        # Split the filename on '_' and extract the first part 'cropped_num1'
        parts = filename.split('_')
        if parts:
            # Further split 'cropped_num1' on '/' to isolate 'num1' and convert it to an integer
            num_parts = parts[0].split('/')
            if len(num_parts) > 1:
                try:
                    return int(num_parts[1])  # Convert 'num1' to integer for sorting
                except ValueError:
                    pass  # In case 'num1' is not a valid integer, ignore this file for sorting
        return float('inf')  # Default value for sorting if 'num1' cannot be extracted

    # Sort the matching files by 'num1'
    matching_files.sort(key=extract_num1)

    # Prepend the directory path to each filename
    matching_files_with_path = [os.path.join(directory, filename) for filename in matching_files]

    return matching_files_with_path


from PIL import Image
import os
import math

def create_square_collage(images, output_path):
    if not images:
        print("No images to create a collage.")
        return

    with Image.open(images[0]) as img:
        width, height = img.size
        num_images = len(images)
        # Calculate the next perfect square number and the number of images per side
        next_perfect_square = math.ceil(math.sqrt(num_images)) ** 2
        num_images_side = int(math.sqrt(next_perfect_square))

        # Calculate how many white images are needed to make the total count a perfect square
        white_images_needed = next_perfect_square - num_images

        # Creating a list to hold actual Image objects (including placeholders)
        image_objects = []

        # Load all actual images first
        for image_path in images:
            with Image.open(image_path) as img:
                image_objects.append(img.copy())

        # Add white placeholders as needed
        for _ in range(white_images_needed):
            white_placeholder = Image.new('RGB', (width, height), color='white')
            image_objects.append(white_placeholder)

        collage_width = width * num_images_side
        collage_height = height * num_images_side

        collage = Image.new('RGB', (collage_width, collage_height), 'white')

        x_offset, y_offset = 0, 0
        for img in image_objects:
            collage.paste(img, (x_offset, y_offset))
            x_offset += width
            if x_offset >= collage_width:
                x_offset = 0
                y_offset += height

    collage.save(output_path)


# Example usage
directory = "./output_images"
word = "WOMENShortsid0000493305_2side_all"
output_path = "testsss.jpg"

# Find images
images_with_word = find_images_with_word(directory, word)
print(len(images_with_word))
# Create collage
create_square_collage(images_with_word, output_path)
