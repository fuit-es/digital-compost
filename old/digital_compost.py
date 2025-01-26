import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from sklearn.decomposition import PCA
from PIL import Image

def load_and_preprocess_image(image_path, resize_dim=(256, 256)):
    """
    Load and resize an image to a manageable size.
    """
    image = Image.open(image_path).convert('RGB')  # Ensure RGB
    image = image.resize(resize_dim, Image.LANCZOS)  # High-quality resize
    return np.array(image)

def compress_pixels(image_array, n_components=50):
    """
    Use PCA to compress the spatial resolution of the image.
    """
    H, W, C = image_array.shape  # Height, Width, Color channels

    # Reshape the image to (H, W * C)
    reshaped_image = image_array.reshape(H, -1)

    # Apply PCA to reduce the number of rows (height)
    pca = PCA(n_components=n_components)
    reduced = pca.fit_transform(reshaped_image)

    # Reconstruct the image from the reduced data
    reconstructed = pca.inverse_transform(reduced)

    # Reshape back to (H, W, C) format
    reconstructed_image = reconstructed.reshape(H, W, C).astype(np.uint8)

    return reconstructed_image

def visualize_images(original, compressed):
    """
    Display the original and compressed images side by side.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    
    axes[0].imshow(original)
    axes[0].set_title("Original Image")
    axes[0].axis('off')

    axes[1].imshow(compressed)
    axes[1].set_title("Compressed Image (Reduced Resolution)")
    axes[1].axis('off')

    plt.show()

def prepare_pixel_data(image_array):
    """
    Prepare pixel data for animation: positions and colors.
    """
    H, W, C = image_array.shape

    # Create (x, y) coordinates for all pixels and cast to float64
    y_indices, x_indices = np.indices((H, W))
    positions = np.column_stack((x_indices.flatten(), y_indices.flatten())).astype(np.float64)

    # Get corresponding colors for each pixel
    colors = image_array.reshape(-1, 3) / 255.0  # Normalize RGB to [0, 1]

    return positions, colors

def animate_dust_particles(positions, colors, H, W, num_frames=1000, interval=50):
    """
    Animate the pixel movements like dust particles.
    """
    fig, ax = plt.subplots(figsize=(6, 6))

    # Set up the plot limits and invert the y-axis
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.invert_yaxis()
    ax.axis('off')

    # Create the scatter plot
    scatter = ax.scatter(positions[:, 0], positions[:, 1], c=colors, s=2, edgecolor='none')

    # Initialize velocities for each particle
    velocities = np.random.uniform(-1, 1, size=positions.shape)

    # Function to update velocities to simulate dust particle motion
    def update_velocities(velocities):
        # Random acceleration
        acceleration = np.random.uniform(-0.2, 0.2, size=positions.shape)
        velocities += acceleration
        # Damping factor to prevent velocities from becoming too large
        velocities *= 0.95
        return velocities

    def animate(frame):
        nonlocal positions, velocities
        velocities = update_velocities(velocities)
        positions += velocities  # Update positions

        # Handle boundary conditions (particles wrap around)
        positions[:, 0] = positions[:, 0] % W
        positions[:, 1] = positions[:, 1] % H

        scatter.set_offsets(positions)  # Update scatter plot positions
        return scatter,

    # Use FuncAnimation for smooth animation
    ani = FuncAnimation(
        fig, animate, frames=num_frames, interval=interval, blit=False, repeat=False
    )

    # Display the animation
    plt.show()

def main():
    # Specify your image path
    image_path = 'image.png'  # Replace with your image path

    # Load and preprocess the image
    original_image = load_and_preprocess_image(image_path)
    print("Original image loaded.")

    # Perform pixel compression using PCA
    n_components = 10  # Adjust as needed for desired compression
    compressed_image = compress_pixels(original_image, n_components=n_components)
    print(f"Image compressed to {n_components} components.")

    # Visualize the original and compressed images
    visualize_images(original_image, compressed_image)

    # Prepare data for animation
    positions, colors = prepare_pixel_data(compressed_image)
    H, W, _ = compressed_image.shape

    # Animate the pixel movement like dust particles
    animate_dust_particles(positions, colors, H, W)

if __name__ == "__main__":
    main()