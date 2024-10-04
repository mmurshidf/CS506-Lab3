import numpy as np
from PIL import Image

# Function to load and preprocess the image
def load_image(image_path):
    """
    Loads an image from the specified path and converts it to a NumPy array.

    Args:
        image_path (str): Path to the image file.

    Returns:
        np.ndarray: NumPy array representation of the image.
    """
    try:
        # Open the image using Pillow
        with Image.open(image_path) as img:
            # Convert image to RGB if it's not already in that mode
            if img.mode != 'RGB' and img.mode != 'L':
                img = img.convert('RGB')
            # Convert the image to a NumPy array
            image_np = np.array(img)
        return image_np
    except Exception as e:
        print(f"Error loading image: {e}")
        raise

# Function to perform SVD on a single channel of the image matrix
def compress_channel_svd(channel_matrix, rank):
    """
    Compresses a single color channel of an image using SVD.

    Args:
        channel_matrix (np.ndarray): 2D NumPy array representing a single color channel.
        rank (int): Number of singular values to retain.

    Returns:
        np.ndarray: Compressed color channel as a 2D NumPy array.
    """
    # Perform Singular Value Decomposition
    U, S, Vt = np.linalg.svd(channel_matrix, full_matrices=False)
    
    # Retain only the top 'rank' singular values
    U_r = U[:, :rank]
    S_r = S[:rank]
    Vt_r = Vt[:rank, :]
    
    # Reconstruct the compressed channel matrix
    compressed_channel = np.dot(U_r, np.dot(np.diag(S_r), Vt_r))
    
    return compressed_channel

# Function to perform SVD for image compression
def image_compression_svd(image_np, rank):
    """
    Compresses an image using SVD by applying it to each color channel.

    Args:
        image_np (np.ndarray): NumPy array representation of the image.
        rank (int): Number of singular values to retain for each channel.

    Returns:
        np.ndarray: Compressed image as a NumPy array.
    """
    # Check if the image is grayscale or color image
    if len(image_np.shape) == 2:  # Grayscale
        compressed_img = compress_channel_svd(image_np, rank)
    else:
        # List to store compressed channels
        compressed_channels = []
        
        # Loop over the 3 color channels (RGB)
        for i in range(3):
            channel = image_np[:, :, i]
            compressed_channel = compress_channel_svd(channel, rank)
            compressed_channels.append(compressed_channel)
        
        # Stack the compressed channels back into an RGB image
        compressed_img = np.stack(compressed_channels, axis=2)
        
    # Clip values to ensure they remain in the valid pixel range [0, 255]
    compressed_img = np.clip(compressed_img, 0, 255)
    
    return compressed_img.astype(np.uint8)

# Function to concatenate and save the original and quantized images side by side
def save_result(original_image_np, quantized_image_np, output_path):
    """
    Saves the original and compressed images side by side in a single image file.

    Args:
        original_image_np (np.ndarray): NumPy array of the original image.
        quantized_image_np (np.ndarray): NumPy array of the compressed image.
        output_path (str): Path where the combined image will be saved.
    """
    # Convert NumPy arrays back to PIL images
    original_image = Image.fromarray(original_image_np)
    quantized_image = Image.fromarray(quantized_image_np)
    
    # Get dimensions
    width, height = original_image.size
    
    # Determine the mode based on the image type
    if original_image.mode == 'RGB' and quantized_image.mode == 'RGB':
        combined_mode = 'RGB'
    else:
        combined_mode = 'L'  # Grayscale
    
    # Create a new image that will hold both the original and quantized images side by side
    combined_image = Image.new(combined_mode, (width * 2, height))
    
    # Paste original and quantized images side by side
    combined_image.paste(original_image, (0, 0))
    combined_image.paste(quantized_image, (width, 0))
    
    # Save the combined image
    combined_image.save(output_path)
    print(f"Combined image saved to {output_path}")

if __name__ == '__main__':
    # Load and process the image
    image_path = 'nano.jpeg'  # Replace with your image path
    output_path = 'nano_compressed_image.png'  # Output path for the result
    image_np = load_image(image_path)

    # Perform image compression using SVD
    rank = 12  # Rank for SVD, you may change this to experiment
    quantized_image_np = image_compression_svd(image_np, rank)

    # Save the original and quantized images side by side
    save_result(image_np, quantized_image_np, output_path)
