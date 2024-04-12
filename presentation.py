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

# # Example usage:
# # Define your input and output directories
# input_directory = "path/to/your/input/images"
# output_directory = "path/to/your/output/images"
#
# # Iterate over all images in the input directory
# for filename in os.listdir(input_directory):
#     if filename.endswith(".jpg"):  # Assuming the images are in JPG format; adjust if necessary
#         input_path = os.path.join(input_directory, filename)
#         output_path = os.path.join(output_directory, f"cropped_{filename}")
#         crop_center_square(input_path, output_path)
