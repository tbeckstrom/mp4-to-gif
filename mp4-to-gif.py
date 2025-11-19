import os
import argparse
import numpy as np
from moviepy import VideoFileClip
from PIL import Image, ImageDraw

def add_rounded_corners(frame: np.ndarray, corner_radius: int) -> np.ndarray:
    """
    Applies rounded corners to a single video frame.

    Args:
        frame: The input video frame as a NumPy array.
        corner_radius: The radius for the rounded corners in pixels.

    Returns:
        The frame with rounded corners as a NumPy array.
    """
    # Convert the numpy array frame to a Pillow Image
    image = Image.fromarray(frame).convert("RGBA")

    # Create a mask with the same size as the image, initially all transparent
    mask = Image.new('L', image.size, 0)
    draw = ImageDraw.Draw(mask)

    # Draw a rounded rectangle on the mask. The white part (255) will be kept.
    draw.rounded_rectangle(
        ((0, 0), image.size),
        radius=corner_radius,
        fill=255
    )

    # Apply the mask to the image's alpha channel
    image.putalpha(mask)

    return np.array(image)

def convert_mp4_to_gif_with_rounded_corners(
    input_path: str,
    output_path: str,
    corner_percentage: float = 5.0,
    fps: int = None
):
    """
    Converts an MP4 video to a smooth GIF in its native resolution
    with customizable rounded corners.

    Args:
        input_path: Path to the input MP4 video file.
        output_path: Path to save the output GIF file.
        corner_percentage: The percentage of the smaller dimension (width or height)
                           to use for the corner radius.
        fps: The frames per second for the output GIF. If None, uses the
             source video's FPS for maximum smoothness.
    """
    if not os.path.exists(input_path):
        print(f"Error: Input file not found at '{input_path}'")
        return

    print("Processing started...")

    # Load the video clip using MoviePy
    with VideoFileClip(input_path) as clip:
        # Get native resolution and FPS
        width, height = clip.size
        native_fps = clip.fps
        
        print(f"Video detected with resolution {width}x{height} and {native_fps:.2f} FPS.")

        # Calculate the corner radius in pixels from the percentage
        radius_pixels = int((corner_percentage / 100.0) * min(width, height))
        print(f"Applying rounded corners with a radius of {radius_pixels} pixels.")

        # Define a function to apply the rounded corners to each frame
        def process_frame(frame):
            return add_rounded_corners(frame, radius_pixels)

        # Apply the transformation to each frame of the clip
        rounded_clip = clip.image_transform(process_frame)

        # Use the video's native FPS for a smooth GIF, or the user-specified one
        output_fps = fps if fps is not None else native_fps

        # Write the final result to a GIF file
        print(f"Generating GIF with {output_fps:.2f} FPS...")
        rounded_clip.write_gif(output_path, fps=output_fps)

    print(f"\nSuccessfully converted video and saved GIF to '{output_path}'")


if __name__ == '__main__':
    # --- Command-Line Argument Parsing ---
    parser = argparse.ArgumentParser(
        description="Convert an MP4 video to a smooth GIF with rounded corners.",
        formatter_class=argparse.RawTextHelpFormatter # For better help text formatting
    )

    # Positional (required) arguments
    parser.add_argument(
        "input_path",
        type=str,
        help="Path to the input MP4 video file."
    )
    parser.add_argument(
        "output_path",
        type=str,
        help="Path to save the output GIF file."
    )

    # Optional arguments
    parser.add_argument(
        "-c", "--corner-percentage",
        type=float,
        default=5.0,
        help="Percentage of the smaller dimension for the corner radius.\n(default: 5.0)"
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=None,
        help="Frames per second for the output GIF.\n(default: use source video's FPS)"
    )

    # Parse the arguments from the command line
    args = parser.parse_args()

    # --- Run the Conversion ---
    convert_mp4_to_gif_with_rounded_corners(
        input_path=args.input_path,
        output_path=args.output_path,
        corner_percentage=args.corner_percentage,
        fps=args.fps
    )