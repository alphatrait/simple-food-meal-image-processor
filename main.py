from rembg import remove
import cv2
import numpy as np
from PIL import Image
import os
from PIL import ImageEnhance
from storage import save_in_gcs

# Define the folder containing the input images
input_folder = 'images'

# Get a list of all files in the input folder
input_files = os.listdir(input_folder)

# Initialize a variable to keep track of the image number
image_number = 0


def rotate_if_needed(input):
    # Open the image
    image = Image.open(input)

    # Check if the width is greater than the height
    if image.width > image.height:
        # Rotate the image 90 degrees to the right
        rotated_image = image.transpose(Image.ROTATE_270)
        rotated_image.save("temp/ready.jpg")
        print(f"Rotated {input} 90 degrees to the right.")
    else:
        # No rotation needed, just save the original image
        image.save("temp/ready.jpg")
        print(f"No rotation needed for {input}.")

    return "temp/ready.jpg"



def enhance_image(input):
    # Open the image
    image = Image.open(input)

    # Define enhancement factors
    brightness_factor = 1.03  # Increase brightness by 20%
    contrast_factor = 1.05    # Increase contrast by 20%
    highlight_factor = 1.05  # Increase highlight by 20%
    shadows_factor = 1.05    # Increase shadows by 20% (lighter)
    saturation_factor = 1.07 # Increase saturation by 20%
    vibrance_factor = 1.05   # Increase vibrance by 20%
    sharpness_factor = 1.05  # Increase sharpness by 20%

    # Apply enhancements
    enhancer = ImageEnhance.Brightness(image)
    image = enhancer.enhance(brightness_factor)

    enhancer = ImageEnhance.Contrast(image)
    image = enhancer.enhance(contrast_factor)

    enhancer = ImageEnhance.Brightness(image)
    image = enhancer.enhance(highlight_factor)

    enhancer = ImageEnhance.Brightness(image)
    image = enhancer.enhance(shadows_factor)

    enhancer = ImageEnhance.Color(image)
    image = enhancer.enhance(saturation_factor)

    enhancer = ImageEnhance.Color(image)
    image = enhancer.enhance(vibrance_factor)

    enhancer = ImageEnhance.Sharpness(image)
    image = enhancer.enhance(sharpness_factor)

    # Save the enhanced image
    image.save("temp/enhanced.jpg")
    print(f"Enhanced the meal image and saved.")
    
    return "temp/enhanced.jpg"



def square_the_image(input):
    # Load the image
    image = cv2.imread(input, cv2.IMREAD_UNCHANGED)  # Replace 'your_image.jpg' with the path to your image file

    # Determine the new dimensions for the square image (1080x1080 pixels)
    new_size = (1080, 1080)

    # Calculate the scaling factor to fit the image within the square canvas
    scale = min(new_size[0] / image.shape[0], new_size[1] / image.shape[1])

    # Calculate the new dimensions for the resized image
    resized_width = int(image.shape[1] * scale)
    resized_height = int(image.shape[0] * scale)

    # Resize the image while preserving its transparency
    resized_image = cv2.resize(image, (resized_width, resized_height), interpolation=cv2.INTER_AREA)

    # Create a blank square image with the desired size and an alpha channel
    square_image = np.zeros((new_size[0], new_size[1], 4), dtype=np.uint8)

    # Calculate the position to paste the resized image in the center
    x_offset = (new_size[0] - resized_image.shape[1]) // 2
    y_offset = (new_size[1] - resized_image.shape[0]) // 2

    # Paste the resized image onto the square image, preserving transparency
    square_image[y_offset:y_offset + resized_image.shape[0], x_offset:x_offset + resized_image.shape[1]] = resized_image


    # Save the resulting square image
    cv2.imwrite('temp/square.png', square_image)  # Replace 'output_square_image.jpg' with your desired output file name
    print("Image squared and saved.")

    return 'temp/square.png'



def calculate_and_resize(input):

    # Desired scale relative to the canvas
    desired_scale = 0.6 

    # Load the transparent image
    image = cv2.imread(input, cv2.IMREAD_UNCHANGED)

    if image is None:
        print("Failed to load the image.")
        return

    # Find the coordinates of the object within the transparent image
    object_coords = np.argwhere(image[:, :, 3] > 0)
    (min_y, min_x), (max_y, max_x) = object_coords.min(0), object_coords.max(0)

    # Calculate the width and height of the object
    object_width = max_x - min_x + 1
    object_height = max_y - min_y + 1

    # Calculate the dimensions of the canvas
    canvas_height, canvas_width, _ = image.shape

    # Calculate the current scale of the object relative to the canvas
    current_scale_x = object_width / canvas_width
    current_scale_y = object_height / canvas_height

    # Calculate the desired scale factor to make the object 0.6 relative to the canvas
    scale_factor_x = desired_scale / current_scale_x
    scale_factor_y = desired_scale / current_scale_y

    # Resize the object using OpenCV
    resized_object = cv2.resize(image[min_y:max_y + 1, min_x:max_x + 1], None, fx=scale_factor_x, fy=scale_factor_y)

    # Create a new canvas of the same size as the original image
    canvas = np.zeros_like(image)

    # Place the resized object back onto the canvas
    canvas_height, canvas_width, _ = canvas.shape
    new_object_height, new_object_width, _ = resized_object.shape
    y_offset = (canvas_height - new_object_height) // 2
    x_offset = (canvas_width - new_object_width) // 2
    canvas[y_offset:y_offset + new_object_height, x_offset:x_offset + new_object_width] = resized_object

    # Save the modified image
    cv2.imwrite("temp/scale.png", canvas)
    print(f"Scale adjusted and saved.")

    return "temp/scale.png"



def align_center_of_object(image_path):
    # Load the image with an alpha channel
    image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)

    if image is None:
        print("Failed to load the image.")
        return None

    # Extract the alpha channel
    alpha_channel = image[:, :, 3]

    # Find non-transparent regions (where alpha is not 0)
    non_transparent_mask = alpha_channel > 0

    # Find contours of non-transparent regions
    contours, _ = cv2.findContours(non_transparent_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        print("No non-transparent regions found.")
        return None

    # Assuming there is only one contour, calculate its centroid (center of mass)
    largest_contour = max(contours, key=cv2.contourArea)
    moments = cv2.moments(largest_contour)
    centroid_x = int(moments['m10'] / moments['m00'])
    centroid_y = int(moments['m01'] / moments['m00'])

    # Get the dimensions of the image
    image_height, image_width = image.shape[:2]

    # Calculate the translation needed to align the centroid with the center of the canvas
    translate_x = (image_width // 2) - centroid_x
    translate_y = (image_height // 2) - centroid_y

    # Create an empty canvas with the same dimensions as the original image
    canvas = np.zeros_like(image)

    # Translate the image to align the centroid with the center of the canvas
    translation_matrix = np.float32([[1, 0, translate_x], [0, 1, translate_y]])
    translated_image = cv2.warpAffine(image, translation_matrix, (image_width, image_height))

    # Copy the translated image to the canvas
    canvas = canvas + translated_image

    cv2.imwrite('temp/centered.png', canvas)
    print("Center of the meal aligned with the center of the canvas and saved.")

    return 'temp/centered.png'



def remove_bg(input):
    input_image = cv2.imread(input)
    output_image = remove(input_image)
    cv2.imwrite("temp/output.png", output_image)
    print("Background removed and saved.")

    return "temp/output.png"



def rotate_image(input):
    img = cv2.imread(input, cv2.IMREAD_UNCHANGED)
    height, width = img.shape[:2]
    rotation_matrix = cv2.getRotationMatrix2D((width / 2, height / 2), -45, 1)
    img_rotation = cv2.warpAffine(img, rotation_matrix, (width, height), flags=cv2.INTER_AREA)
    
    cv2.imwrite('temp/rotated.png', img_rotation) 
    print("Image rotated and saved.")

    return 'temp/rotated.png'



def blur_image(input):
    image = cv2.imread(input, cv2.IMREAD_UNCHANGED)
    
    # Separate the alpha channel
    rgb_channels = image[:, :, :3]
    alpha_channel = image[:, :, 3]

    # Create a mask for the non-transparent areas
    _, mask = cv2.threshold(alpha_channel, 0, 255, cv2.THRESH_BINARY)

    # Create a larger mask for the blurred area
    kernel = np.ones((41, 41), np.uint8)
    dilated_mask = cv2.dilate(mask, kernel, iterations=1)

    # Invert the masks
    inv_mask = cv2.bitwise_not(mask)
    inv_dilated_mask = cv2.bitwise_not(dilated_mask)

    # Create the blur effect
    blurred_rgb = cv2.GaussianBlur(rgb_channels, (41, 41), 0)

    # Combine the original image with the blurred edges
    result_rgb = rgb_channels.copy()
    result_rgb = cv2.bitwise_and(result_rgb, result_rgb, mask=mask)
    blurred_edges = cv2.bitwise_and(blurred_rgb, blurred_rgb, mask=inv_dilated_mask)
    result_rgb = cv2.add(result_rgb, blurred_edges)

    # Combine the result with the original alpha channel
    result = cv2.merge([result_rgb, alpha_channel])

    cv2.imwrite('temp/blur.png', result)
    print("Image blurred and saved.")

    return 'temp/blur.png'



def add(input, name):
    background = Image.open("assets/bg.png")
    foreground = Image.open(input)
    foreground_with_alpha = foreground.convert("RGBA")

    final = Image.new("RGBA", background.size)
    final = Image.alpha_composite(final, background)
    final = Image.alpha_composite(final, foreground_with_alpha)

    final.save("edited/" + name + ".png")
    print("Image saved to edited folder.")



def meal_image_editor(input, name):
    try:
        input = rotate_if_needed(input)
        if not input:
            raise ValueError("Rotation failed")

        input = enhance_image(input)
        if not input:
            raise ValueError("Enhancement failed")

        input = remove_bg(input)
        if not input:
            raise ValueError("Background removal failed")

        input = calculate_and_resize(input)
        if not input:
            raise ValueError("Resizing failed")

        input = align_center_of_object(input)
        if not input:
            raise ValueError("Centering failed")

        input = rotate_image(input)
        if not input:
            raise ValueError("Rotation failed")

        input = square_the_image(input)
        if not input:
            raise ValueError("Squaring failed")

        input = blur_image(input)
        if not input:
            raise ValueError("Blurring failed")

        add(input, name)
        
        return save_in_gcs(f"edited/{name}.png")
    except Exception as e:
        print(f"Error in meal_image_editor: {str(e)}")
        return None
