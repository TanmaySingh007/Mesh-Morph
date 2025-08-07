"""
Utility functions for texture processing and enhancement.
"""

import os
import hashlib
import json
from typing import List, Dict, Optional, Tuple
from PIL import Image, ImageFilter, ImageEnhance
import numpy as np
from datetime import datetime

def create_seamless_texture(image: Image.Image, tile_size: int = 256) -> Image.Image:
    """
    Create a seamless texture by tiling and blending the image.
    
    Args:
        image: Input image to make seamless
        tile_size: Size of the tile for seamless generation
        
    Returns:
        Seamless texture image
    """
    # Resize image to tile size
    image = image.resize((tile_size, tile_size), Image.Resampling.LANCZOS)
    
    # Create a larger canvas
    canvas_size = tile_size * 2
    canvas = Image.new('RGB', (canvas_size, canvas_size))
    
    # Tile the image
    for y in range(2):
        for x in range(2):
            canvas.paste(image, (x * tile_size, y * tile_size))
    
    # Apply blur to smooth edges
    canvas = canvas.filter(ImageFilter.GaussianBlur(radius=1))
    
    # Crop back to original size
    seamless = canvas.crop((tile_size//2, tile_size//2, 
                           tile_size//2 + tile_size, tile_size//2 + tile_size))
    
    return seamless

def enhance_texture(image: Image.Image, 
                   contrast: float = 1.2,
                   brightness: float = 1.1,
                   sharpness: float = 1.3) -> Image.Image:
    """
    Enhance texture quality using various image adjustments.
    
    Args:
        image: Input image
        contrast: Contrast adjustment factor
        brightness: Brightness adjustment factor
        sharpness: Sharpness adjustment factor
        
    Returns:
        Enhanced image
    """
    # Apply contrast enhancement
    enhancer = ImageEnhance.Contrast(image)
    image = enhancer.enhance(contrast)
    
    # Apply brightness enhancement
    enhancer = ImageEnhance.Brightness(image)
    image = enhancer.enhance(brightness)
    
    # Apply sharpness enhancement
    enhancer = ImageEnhance.Sharpness(image)
    image = enhancer.enhance(sharpness)
    
    return image

def generate_filename(prompt: str, timestamp: bool = True) -> str:
    """
    Generate a filename based on the prompt.
    
    Args:
        prompt: Text prompt used for generation
        timestamp: Whether to include timestamp
        
    Returns:
        Generated filename
    """
    # Clean the prompt for filename
    clean_prompt = "".join(c for c in prompt if c.isalnum() or c in (' ', '-', '_')).rstrip()
    clean_prompt = clean_prompt.replace(' ', '_')[:50]  # Limit length
    
    # Create hash for uniqueness
    prompt_hash = hashlib.md5(prompt.encode()).hexdigest()[:8]
    
    filename = f"{clean_prompt}_{prompt_hash}"
    
    if timestamp:
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{filename}_{timestamp_str}"
    
    return f"{filename}.png"

def save_texture_with_metadata(image: Image.Image, 
                              filename: str, 
                              prompt: str,
                              generation_params: Dict,
                              output_dir: str = "generated_textures") -> Tuple[str, str]:
    """
    Save texture with metadata.
    
    Args:
        image: Image to save
        filename: Base filename
        prompt: Generation prompt
        generation_params: Parameters used for generation
        output_dir: Output directory
        
    Returns:
        Tuple of (image_path, metadata_path)
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Save image
    image_path = os.path.join(output_dir, filename)
    image.save(image_path, "PNG")
    
    # Save metadata
    metadata = {
        "prompt": prompt,
        "generation_params": generation_params,
        "filename": filename,
        "created_at": datetime.now().isoformat(),
        "image_size": image.size,
        "image_mode": image.mode
    }
    
    metadata_filename = filename.replace('.png', '_metadata.json')
    metadata_path = os.path.join(output_dir, metadata_filename)
    
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    return image_path, metadata_path

def load_texture_metadata(metadata_path: str) -> Dict:
    """
    Load texture metadata from file.
    
    Args:
        metadata_path: Path to metadata file
        
    Returns:
        Metadata dictionary
    """
    with open(metadata_path, 'r') as f:
        return json.load(f)

def batch_generate_textures(generator, prompts: List[str], 
                           output_dir: str = "generated_textures",
                           enhance: bool = True) -> List[str]:
    """
    Generate multiple textures in batch.
    
    Args:
        generator: TextureGenerator instance
        prompts: List of prompts to generate
        output_dir: Output directory
        enhance: Whether to enhance textures
        
    Returns:
        List of generated file paths
    """
    generated_files = []
    
    for i, prompt in enumerate(prompts):
        try:
            # Generate texture
            texture = generator.generate_texture(prompt)
            
            # Enhance if requested
            if enhance:
                texture = enhance_texture(texture)
            
            # Generate filename
            filename = generate_filename(prompt)
            
            # Save with metadata
            generation_params = {
                "width": texture.width,
                "height": texture.height,
                "enhanced": enhance
            }
            
            image_path, metadata_path = save_texture_with_metadata(
                texture, filename, prompt, generation_params, output_dir
            )
            
            generated_files.append(image_path)
            print(f"Generated texture {i+1}/{len(prompts)}: {filename}")
            
        except Exception as e:
            print(f"Failed to generate texture {i+1}: {e}")
    
    return generated_files

def create_texture_variations(base_image: Image.Image, 
                            num_variations: int = 4,
                            variation_strength: float = 0.3) -> List[Image.Image]:
    """
    Create variations of a base texture.
    
    Args:
        base_image: Base texture image
        num_variations: Number of variations to create
        variation_strength: Strength of variations (0.0 to 1.0)
        
    Returns:
        List of variation images
    """
    variations = []
    
    for i in range(num_variations):
        # Convert to numpy array
        img_array = np.array(base_image)
        
        # Add random noise
        noise = np.random.normal(0, variation_strength * 50, img_array.shape)
        img_array = np.clip(img_array + noise, 0, 255).astype(np.uint8)
        
        # Convert back to PIL Image
        variation = Image.fromarray(img_array)
        
        # Apply slight color variations
        enhancer = ImageEnhance.Color(variation)
        variation = enhancer.enhance(1.0 + (np.random.random() - 0.5) * variation_strength)
        
        variations.append(variation)
    
    return variations

def validate_image_dimensions(width: int, height: int) -> Tuple[int, int]:
    """
    Validate and adjust image dimensions to be compatible with the model.
    
    Args:
        width: Requested width
        height: Requested height
        
    Returns:
        Tuple of (validated_width, validated_height)
    """
    # Ensure dimensions are multiples of 8 (required by Stable Diffusion)
    width = (width // 8) * 8
    height = (height // 8) * 8
    
    # Set minimum and maximum limits
    min_size = 256
    max_size = 1024
    
    width = max(min_size, min(max_size, width))
    height = max(min_size, min(max_size, height))
    
    return width, height 