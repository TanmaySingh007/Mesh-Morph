import torch
from diffusers import StableDiffusionPipeline
from PIL import Image
import logging
import os
from typing import List
from config import (
    DEFAULT_MODEL_CONFIG, 
    DEFAULT_GENERATION_CONFIG, 
    DEFAULT_OUTPUT_CONFIG,
    DEFAULT_SYSTEM_CONFIG,
    TEXTURE_PROMPTS,
    GENERATION_PRESETS
)
from utils import (
    enhance_texture,
    generate_filename,
    save_texture_with_metadata,
    batch_generate_textures,
    create_seamless_texture,
    validate_image_dimensions
)
from utils_3d import (
    create_textured_cube,
    create_textured_sphere,
    create_textured_cylinder
)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TextureGenerator:
    def __init__(self, model_config=None, generation_config=None, system_config=None):
        """
        Initialize the texture generator with a pre-trained diffusion model.
        
        Args:
            model_config: Model configuration (uses default if None)
            generation_config: Generation configuration (uses default if None)
            system_config: System configuration (uses default if None)
        """
        self.model_config = model_config or DEFAULT_MODEL_CONFIG
        self.generation_config = generation_config or DEFAULT_GENERATION_CONFIG
        self.system_config = system_config or DEFAULT_SYSTEM_CONFIG
        
        # Determine device
        if self.system_config.device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = self.system_config.device
            
        self.pipeline = None
        logger.info(f"Using device: {self.device}")
        
    def load_model(self):
        """Load the pre-trained diffusion model."""
        try:
            logger.info(f"Loading model: {self.model_config.model_id}")
            
            # Determine torch dtype
            torch_dtype = torch.float16 if self.device == "cuda" else torch.float32
            
            self.pipeline = StableDiffusionPipeline.from_pretrained(
                self.model_config.model_id,
                torch_dtype=torch_dtype,
                safety_checker=self.model_config.safety_checker,
                requires_safety_checker=self.model_config.requires_safety_checker
            )
            
            if self.device == "cuda":
                self.pipeline = self.pipeline.to(self.device)
                
                # Enable memory optimizations
                if self.generation_config.enable_attention_slicing and hasattr(self.pipeline, "enable_attention_slicing"):
                    self.pipeline.enable_attention_slicing()
                if self.generation_config.enable_vae_slicing and hasattr(self.pipeline, "enable_vae_slicing"):
                    self.pipeline.enable_vae_slicing()
                if self.generation_config.enable_memory_efficient_attention:
                    # Enable memory efficient attention if available
                    pass  # This is handled automatically by diffusers
            
            logger.info("Model loaded successfully")
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise
    
    def generate_texture(self, prompt: str, width: int = None, height: int = None, 
                        num_inference_steps: int = None, guidance_scale: float = None,
                        seed: int = None, enhance: bool = True, seamless: bool = False) -> Image.Image:
        """
        Generate a texture image from a text prompt.
        
        Args:
            prompt (str): Text description of the texture to generate
            width (int): Width of the output image (uses default if None)
            height (int): Height of the output image (uses default if None)
            num_inference_steps (int): Number of denoising steps (uses default if None)
            guidance_scale (float): Guidance scale for classifier-free guidance (uses default if None)
            seed (int): Random seed for reproducibility (uses default if None)
            enhance (bool): Whether to enhance the texture after generation
            seamless (bool): Whether to make the texture seamless
            
        Returns:
            PIL.Image.Image: Generated texture image
        """
        if self.pipeline is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        # Use smart texture generation by default
        return self.generate_smart_texture(prompt, width=width, height=height, 
                                         num_inference_steps=num_inference_steps,
                                         guidance_scale=guidance_scale, seed=seed,
                                         enhance=enhance, seamless=seamless)
    
    def save_texture(self, image: Image.Image, prompt: str, filename: str = None, 
                    output_dir: str = None, save_metadata: bool = True):
        """
        Save the generated texture to a file with optional metadata.
        
        Args:
            image (PIL.Image.Image): The image to save
            prompt (str): The prompt used to generate the texture
            filename (str): Output filename (auto-generated if None)
            output_dir (str): Output directory (uses default if None)
            save_metadata (bool): Whether to save metadata file
        """
        try:
            if filename is None:
                filename = generate_filename(prompt)
            
            output_dir = output_dir or DEFAULT_OUTPUT_CONFIG.output_dir
            
            if save_metadata:
                generation_params = {
                    "width": image.width,
                    "height": image.height,
                    "device": self.device,
                    "model_id": self.model_config.model_id
                }
                
                image_path, metadata_path = save_texture_with_metadata(
                    image, filename, prompt, generation_params, output_dir
                )
                logger.info(f"Texture and metadata saved: {image_path}")
                return image_path, metadata_path
            else:
                # Simple save without metadata
                os.makedirs(output_dir, exist_ok=True)
                image_path = os.path.join(output_dir, filename)
                image.save(image_path, "PNG")
                logger.info(f"Texture saved: {image_path}")
                return image_path, None
                
        except Exception as e:
            logger.error(f"Error saving texture: {e}")
            raise
    
    def generate_with_preset(self, prompt: str, preset: str = "quality", **kwargs):
        """
        Generate texture using a predefined preset.
        
        Args:
            prompt (str): Text prompt
            preset (str): Preset name ('fast', 'quality', 'detailed')
            **kwargs: Additional parameters to override preset
            
        Returns:
            PIL.Image.Image: Generated texture
        """
        if preset not in GENERATION_PRESETS:
            raise ValueError(f"Unknown preset: {preset}. Available: {list(GENERATION_PRESETS.keys())}")
        
        preset_params = GENERATION_PRESETS[preset].copy()
        preset_params.update(kwargs)
        
        return self.generate_texture(prompt, **preset_params)
    
    def generate_material_texture(self, material: str, style: str = "default", **kwargs):
        """
        Generate texture for a specific material using predefined prompts.
        
        Args:
            material (str): Material type (e.g., 'wood', 'stone', 'metal', 'fabric', 'leather', 'plastic', 'glass', 'ceramic', etc.)
            style (str): Style variant ('default', '0', '1', '2', etc.)
            **kwargs: Additional generation parameters
            
        Returns:
            PIL.Image.Image: Generated texture
        """
        # Normalize material name
        material = material.lower().strip()
        
        # Map common variations to standard materials
        material_mapping = {
            'wooden': 'wood',
            'wooden': 'wood',
            'oak': 'wood',
            'pine': 'wood',
            'mahogany': 'wood',
            'stone': 'stone',
            'rock': 'stone',
            'granite': 'granite',
            'marble': 'marble',
            'slate': 'slate',
            'limestone': 'limestone',
            'sandstone': 'sandstone',
            'basalt': 'basalt',
            'metal': 'metal',
            'steel': 'metal',
            'aluminum': 'metal',
            'copper': 'metal',
            'iron': 'metal',
            'chrome': 'metal',
            'fabric': 'fabric',
            'cloth': 'fabric',
            'cotton': 'fabric',
            'silk': 'fabric',
            'wool': 'fabric',
            'linen': 'fabric',
            'denim': 'fabric',
            'leather': 'leather',
            'suede': 'leather',
            'plastic': 'plastic',
            'glass': 'glass',
            'ceramic': 'ceramic',
            'porcelain': 'ceramic',
            'terracotta': 'ceramic',
            'concrete': 'concrete',
            'cement': 'concrete',
            'brick': 'brick',
            'tile': 'tile',
            'carpet': 'carpet',
            'rug': 'carpet',
            'paper': 'paper',
            'cardboard': 'paper',
            'parchment': 'paper',
            'rubber': 'rubber',
            'cork': 'cork',
            'carpet': 'carpet'
        }
        
        # Use mapping if available, otherwise use the original material
        material = material_mapping.get(material, material)
        
        if material not in TEXTURE_PROMPTS:
            # If material not found, try to find a similar one
            available_materials = list(TEXTURE_PROMPTS.keys())
            logger.warning(f"Unknown material: {material}. Available: {available_materials}")
            
            # Try to find a similar material
            for available in available_materials:
                if material in available or available in material:
                    material = available
                    logger.info(f"Using similar material: {material}")
                    break
            else:
                # Default to wood if no match found
                material = 'wood'
                logger.info(f"Defaulting to wood material")
        
        prompts = TEXTURE_PROMPTS[material]
        
        if style == "default":
            prompt = prompts[0]
        elif style.isdigit() and 0 <= int(style) < len(prompts):
            prompt = prompts[int(style)]
        else:
            prompt = prompts[0]
        
        logger.info(f"Generating texture for material: {material} with prompt: {prompt}")
        return self.generate_texture_direct(prompt, **kwargs)
    
    def detect_material_from_prompt(self, prompt: str) -> str:
        """
        Detect the material type from a text prompt.
        
        Args:
            prompt (str): Text prompt describing the texture
            
        Returns:
            str: Detected material type
        """
        prompt_lower = prompt.lower()
        
        # Define material keywords
        material_keywords = {
            'wood': ['wood', 'wooden', 'oak', 'pine', 'mahogany', 'grain', 'timber'],
            'stone': ['stone', 'rock', 'granite', 'marble', 'slate', 'limestone', 'sandstone', 'basalt'],
            'metal': ['metal', 'steel', 'aluminum', 'copper', 'iron', 'chrome', 'brushed', 'metallic'],
            'fabric': ['fabric', 'cloth', 'cotton', 'silk', 'wool', 'linen', 'denim', 'woven'],
            'leather': ['leather', 'suede', 'hide', 'skin'],
            'plastic': ['plastic', 'polymer', 'synthetic'],
            'glass': ['glass', 'transparent', 'reflective', 'frosted'],
            'ceramic': ['ceramic', 'porcelain', 'terracotta', 'pottery'],
            'concrete': ['concrete', 'cement', 'aggregate'],
            'brick': ['brick', 'masonry'],
            'tile': ['tile', 'mosaic'],
            'carpet': ['carpet', 'rug', 'berber', 'shag'],
            'paper': ['paper', 'cardboard', 'parchment', 'kraft'],
            'rubber': ['rubber', 'silicone', 'neoprene'],
            'cork': ['cork', 'corkboard']
        }
        
        # Count keyword matches for each material
        material_scores = {}
        for material, keywords in material_keywords.items():
            score = sum(1 for keyword in keywords if keyword in prompt_lower)
            if score > 0:
                material_scores[material] = score
        
        # Return the material with the highest score, or 'wood' as default
        if material_scores:
            best_material = max(material_scores, key=material_scores.get)
            logger.info(f"Detected material: {best_material} from prompt")
            return best_material
        else:
            logger.info("No specific material detected, using default")
            return 'wood'
    
    def generate_smart_texture(self, prompt: str, **kwargs):
        """
        Generate texture with smart material detection.
        
        Args:
            prompt (str): Text description of the texture
            **kwargs: Additional generation parameters
            
        Returns:
            PIL.Image.Image: Generated texture
        """
        # Detect material from prompt
        detected_material = self.detect_material_from_prompt(prompt)
        
        # Generate texture using the detected material
        return self.generate_material_texture(detected_material, **kwargs)
    
    def generate_texture_direct(self, prompt: str, width: int = None, height: int = None, 
                               num_inference_steps: int = None, guidance_scale: float = None,
                               seed: int = None, enhance: bool = True, seamless: bool = False) -> Image.Image:
        """
        Generate a texture image directly from a text prompt without material detection.
        
        Args:
            prompt (str): Text description of the texture to generate
            width (int): Width of the output image (uses default if None)
            height (int): Height of the output image (uses default if None)
            num_inference_steps (int): Number of denoising steps (uses default if None)
            guidance_scale (float): Guidance scale for classifier-free guidance (uses default if None)
            seed (int): Random seed for reproducibility (uses default if None)
            enhance (bool): Whether to enhance the texture after generation
            seamless (bool): Whether to make the texture seamless
            
        Returns:
            PIL.Image.Image: Generated texture image
        """
        if self.pipeline is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        # Use default values if not provided
        width = width or self.generation_config.default_width
        height = height or self.generation_config.default_height
        num_inference_steps = num_inference_steps or self.generation_config.default_inference_steps
        guidance_scale = guidance_scale or self.generation_config.default_guidance_scale
        seed = seed or self.generation_config.default_seed
        
        # Validate dimensions
        width, height = validate_image_dimensions(width, height)
        
        try:
            logger.info(f"Generating texture for prompt: '{prompt}'")
            logger.info(f"Parameters: {width}x{height}, steps={num_inference_steps}, guidance={guidance_scale}")
            
            # Optimize for faster generation
            with torch.no_grad():  # Disable gradient computation for inference
                # Generate the image
                result = self.pipeline(
                    prompt=prompt,
                    width=width,
                    height=height,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale,
                    generator=torch.Generator(device=self.device).manual_seed(seed),
                    output_type="pil"  # Ensure PIL output
                )
            
            # Extract the image from the result
            if hasattr(result, 'images') and len(result.images) > 0:
                image = result.images[0]
            elif hasattr(result, 'image'):
                image = result.image
            else:
                raise RuntimeError("No image generated from pipeline")
            
            # Apply enhancements
            if enhance:
                image = enhance_texture(image)
            
            # Make seamless if requested
            if seamless:
                image = create_seamless_texture(image)
            
            logger.info("Texture generation completed successfully")
            return image
            
        except Exception as e:
            logger.error(f"Error generating texture: {e}")
            # Try with reduced parameters if generation fails
            if num_inference_steps > 15:
                logger.info("Retrying with reduced inference steps...")
                return self.generate_texture_direct(
                    prompt, width, height, 
                    num_inference_steps=15, 
                    guidance_scale=guidance_scale,
                    seed=seed, enhance=enhance, seamless=seamless
                )
            raise
    
    def batch_generate(self, prompts: List[str], **kwargs):
        """
        Generate multiple textures in batch.
        
        Args:
            prompts (List[str]): List of prompts
            **kwargs: Generation parameters
            
        Returns:
            List[str]: List of generated file paths
        """
        return batch_generate_textures(self, prompts, **kwargs)
    
    def generate_3d_asset(self, prompt: str, shape: str = "cube", **kwargs) -> str:
        """
        Generate a 3D asset with texture from a text prompt.
        
        Args:
            prompt (str): Text description of the texture to generate
            shape (str): 3D shape type ('cube', 'sphere', 'cylinder')
            **kwargs: Additional generation parameters
            
        Returns:
            str: Path to the generated .obj file
        """
        # Generate the texture first
        texture = self.generate_texture(prompt, **kwargs)
        
        # Create the 3D asset based on shape
        if shape.lower() == "cube":
            obj_path = create_textured_cube(texture)
        elif shape.lower() == "sphere":
            obj_path = create_textured_sphere(texture)
        elif shape.lower() == "cylinder":
            obj_path = create_textured_cylinder(texture)
        else:
            raise ValueError(f"Unknown shape: {shape}. Available: cube, sphere, cylinder")
        
        logger.info(f"3D asset generated: {obj_path}")
        return obj_path

def main():
    """Example usage of the TextureGenerator."""
    # Initialize the generator
    generator = TextureGenerator()
    
    # Load the model
    generator.load_model()
    
    print("=== Generative AI Texture Model Demo ===\n")
    
    # Example 1: Basic texture generation
    print("1. Generating basic texture...")
    try:
        texture = generator.generate_texture("wood grain texture, detailed, high resolution")
        generator.save_texture(texture, "wood_grain_texture")
        print("✓ Basic texture generated successfully")
    except Exception as e:
        print(f"✗ Failed to generate basic texture: {e}")
    
    # Example 2: Using presets
    print("\n2. Generating texture with quality preset...")
    try:
        texture = generator.generate_with_preset("stone wall texture", preset="quality")
        generator.save_texture(texture, "stone_wall_quality")
        print("✓ Quality preset texture generated successfully")
    except Exception as e:
        print(f"✗ Failed to generate quality preset texture: {e}")
    
    # Example 3: Material-specific generation
    print("\n3. Generating material-specific textures...")
    materials = ["wood", "stone", "metal"]
    for material in materials:
        try:
            texture = generator.generate_material_texture(material, style="default")
            generator.save_texture(texture, f"{material}_texture")
            print(f"✓ {material.capitalize()} texture generated successfully")
        except Exception as e:
            print(f"✗ Failed to generate {material} texture: {e}")
    
    # Example 4: Batch generation
    print("\n4. Batch generating textures...")
    batch_prompts = [
        "fabric texture, soft cotton, woven pattern",
        "leather texture, brown, natural, aged",
        "plastic texture, smooth, glossy surface"
    ]
    
    try:
        generated_files = generator.batch_generate(batch_prompts, enhance=True)
        print(f"✓ Batch generation completed: {len(generated_files)} textures")
    except Exception as e:
        print(f"✗ Batch generation failed: {e}")
    
    # Example 5: Seamless texture
    print("\n5. Generating seamless texture...")
    try:
        texture = generator.generate_texture(
            "seamless tileable texture, brick wall, high resolution",
            seamless=True
        )
        generator.save_texture(texture, "seamless_brick_texture")
        print("✓ Seamless texture generated successfully")
    except Exception as e:
        print(f"✗ Failed to generate seamless texture: {e}")
    
    print("\n=== Demo completed ===")
    print("Check the 'generated_textures' directory for output files.")

if __name__ == "__main__":
    main() 