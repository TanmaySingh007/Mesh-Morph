"""
Configuration settings for the Generative AI Texture Model.
"""

import os
from dataclasses import dataclass
from typing import Optional

@dataclass
class ModelConfig:
    """Configuration for the diffusion model."""
    model_id: str = "runwayml/stable-diffusion-v1-5"
    torch_dtype: str = "float16"  # or "float32" for CPU
    safety_checker: Optional[bool] = None
    requires_safety_checker: bool = False

@dataclass
class GenerationConfig:
    """Configuration for texture generation."""
    default_width: int = 512
    default_height: int = 512
    default_inference_steps: int = 20  # Reduced from 50 for faster generation
    default_guidance_scale: float = 7.5
    default_seed: int = 42
    
    # Advanced settings
    enable_attention_slicing: bool = True
    enable_vae_slicing: bool = True
    enable_memory_efficient_attention: bool = True

@dataclass
class OutputConfig:
    """Configuration for output settings."""
    default_format: str = "PNG"
    quality: int = 95  # For JPEG format
    output_dir: str = "generated_textures"
    
    # Create output directory if it doesn't exist
    def __post_init__(self):
        os.makedirs(self.output_dir, exist_ok=True)

@dataclass
class SystemConfig:
    """System and performance configuration."""
    device: str = "auto"  # "auto", "cuda", "cpu"
    max_memory_usage: Optional[float] = None  # GB
    enable_logging: bool = True
    log_level: str = "INFO"
    output_dir: str = "generated_textures"  # Output directory for generated textures

# Default configurations
DEFAULT_MODEL_CONFIG = ModelConfig()
DEFAULT_GENERATION_CONFIG = GenerationConfig()
DEFAULT_OUTPUT_CONFIG = OutputConfig()
DEFAULT_SYSTEM_CONFIG = SystemConfig()

# Advanced generation presets - optimized for speed
GENERATION_PRESETS = {
    "fast": {
        "num_inference_steps": 15,
        "guidance_scale": 7.0
    },
    "standard": {
        "num_inference_steps": 20,
        "guidance_scale": 7.5
    },
    "quality": {
        "num_inference_steps": 30,
        "guidance_scale": 8.0
    },
    "ultra": {
        "num_inference_steps": 40,
        "guidance_scale": 8.5
    }
}

# Extended texture prompts with more object categories
TEXTURE_PROMPTS = {
    # Original materials
    "wood": [
        "seamless wood grain texture, high resolution, detailed, natural",
        "oak wood texture, smooth grain, professional",
        "rough wood texture, aged, weathered surface",
        "pine wood texture, light color, natural grain",
        "mahogany wood texture, dark, rich, elegant"
    ],
    "stone": [
        "stone wall texture, rough surface, natural, detailed",
        "marble texture, polished, elegant, high resolution",
        "granite texture, speckled, durable surface",
        "limestone texture, porous, natural",
        "slate texture, layered, dark, smooth"
    ],
    "metal": [
        "brushed aluminum texture, industrial, metallic",
        "steel texture, polished, reflective surface",
        "copper texture, aged patina, warm tones",
        "iron texture, rusted, weathered, industrial",
        "chrome texture, highly reflective, smooth"
    ],
    "fabric": [
        "cotton fabric texture, soft, woven pattern",
        "silk texture, smooth, luxurious, flowing",
        "denim texture, blue, rough, durable",
        "wool texture, thick, warm, natural",
        "linen texture, coarse, natural, textured"
    ],
    "leather": [
        "leather texture, brown, natural, aged",
        "suede texture, soft, matte finish",
        "synthetic leather texture, black, smooth",
        "crocodile leather texture, textured, luxury",
        "nappa leather texture, soft, premium"
    ],
    "plastic": [
        "plastic texture, smooth, glossy surface",
        "matte plastic texture, industrial",
        "transparent plastic texture, clear, reflective",
        "textured plastic texture, rough, industrial",
        "colored plastic texture, vibrant, modern"
    ],
    "glass": [
        "glass texture, transparent, reflective, smooth",
        "frosted glass texture, translucent, matte",
        "stained glass texture, colorful, artistic",
        "tempered glass texture, clear, modern",
        "textured glass texture, patterned, decorative"
    ],
    "ceramic": [
        "ceramic texture, smooth, glazed surface, elegant",
        "porcelain texture, white, smooth, delicate",
        "terracotta texture, rough, natural, warm",
        "tile texture, geometric, patterned",
        "pottery texture, handcrafted, natural"
    ],
    "concrete": [
        "concrete texture, rough, industrial, gray",
        "polished concrete texture, smooth, modern",
        "textured concrete texture, patterned, architectural",
        "exposed aggregate concrete texture, rough, natural",
        "stamped concrete texture, decorative, patterned"
    ],
    "brick": [
        "brick texture, red, rough, masonry",
        "white brick texture, clean, modern",
        "aged brick texture, weathered, historical",
        "exposed brick texture, industrial, urban",
        "decorative brick texture, patterned, artistic"
    ],
    "tile": [
        "ceramic tile texture, smooth, glazed",
        "mosaic tile texture, colorful, artistic",
        "stone tile texture, natural, textured",
        "metal tile texture, reflective, modern",
        "glass tile texture, translucent, decorative"
    ],
    "carpet": [
        "carpet texture, soft, plush, colorful",
        "berber carpet texture, looped, textured",
        "oriental rug texture, patterned, traditional",
        "shag carpet texture, deep, luxurious",
        "commercial carpet texture, durable, patterned"
    ],
    "paper": [
        "paper texture, smooth, white, clean",
        "cardboard texture, corrugated, brown",
        "parchment texture, aged, historical",
        "kraft paper texture, brown, natural",
        "textured paper texture, embossed, decorative"
    ],
    "rubber": [
        "rubber texture, smooth, black, industrial",
        "textured rubber texture, grip pattern",
        "foam rubber texture, porous, soft",
        "silicone rubber texture, smooth, flexible",
        "neoprene texture, textured, durable"
    ],
    "cork": [
        "cork texture, natural, porous, brown",
        "cork board texture, textured, functional",
        "cork tile texture, geometric, natural",
        "aged cork texture, weathered, vintage",
        "polished cork texture, smooth, modern"
    ],
    "marble": [
        "white marble texture, polished, elegant",
        "black marble texture, dramatic, luxury",
        "green marble texture, natural, veined",
        "pink marble texture, soft, decorative",
        "travertine marble texture, porous, natural"
    ],
    "granite": [
        "black granite texture, polished, elegant",
        "gray granite texture, speckled, durable",
        "pink granite texture, warm, natural",
        "white granite texture, clean, modern",
        "green granite texture, natural, textured"
    ],
    "slate": [
        "black slate texture, smooth, natural",
        "gray slate texture, layered, textured",
        "green slate texture, natural, rustic",
        "purple slate texture, unique, decorative",
        "textured slate texture, rough, natural"
    ],
    "sandstone": [
        "sandstone texture, porous, natural, warm",
        "red sandstone texture, desert, natural",
        "yellow sandstone texture, weathered, aged",
        "white sandstone texture, clean, natural",
        "textured sandstone texture, rough, natural"
    ],
    "limestone": [
        "limestone texture, porous, natural, light",
        "white limestone texture, clean, smooth",
        "gray limestone texture, natural, textured",
        "fossil limestone texture, detailed, natural",
        "polished limestone texture, smooth, elegant"
    ],
    "basalt": [
        "basalt texture, dark, volcanic, natural",
        "black basalt texture, smooth, modern",
        "textured basalt texture, rough, natural",
        "polished basalt texture, elegant, dark",
        "columnar basalt texture, geometric, natural"
    ],
    
    # New object categories
    "animals": [
        "animal fur texture, soft, natural, detailed",
        "animal skin texture, leathery, natural",
        "animal scales texture, reptilian, detailed",
        "animal feathers texture, soft, colorful",
        "animal hide texture, rough, natural"
    ],
    "cars": [
        "car paint texture, metallic, glossy, modern",
        "car interior texture, leather, luxury",
        "car dashboard texture, plastic, modern",
        "car tire texture, rubber, textured",
        "car chrome texture, reflective, polished"
    ],
    "fruits": [
        "fruit skin texture, natural, colorful",
        "citrus peel texture, rough, textured",
        "apple skin texture, smooth, red, natural",
        "banana peel texture, yellow, natural",
        "grape skin texture, purple, smooth"
    ],
    "buildings": [
        "building wall texture, concrete, industrial",
        "building roof texture, shingle, weathered",
        "building window texture, glass, modern",
        "building door texture, wood, aged",
        "building facade texture, stone, architectural"
    ],
    "nature": [
        "tree bark texture, rough, natural, brown",
        "leaf texture, green, natural, detailed",
        "grass texture, green, natural, soft",
        "rock texture, rough, natural, gray",
        "soil texture, brown, natural, textured"
    ],
    "food": [
        "bread texture, golden, crusty, natural",
        "cheese texture, yellow, aged, natural",
        "meat texture, red, natural, detailed",
        "vegetable texture, green, natural, fresh",
        "grain texture, natural, brown, textured"
    ],
    "electronics": [
        "circuit board texture, green, detailed",
        "metal casing texture, aluminum, modern",
        "plastic housing texture, smooth, modern",
        "screen texture, glass, reflective",
        "button texture, plastic, tactile"
    ],
    "clothing": [
        "cotton fabric texture, soft, natural",
        "denim texture, blue, rough, durable",
        "silk texture, smooth, luxurious",
        "wool texture, thick, warm, natural",
        "leather texture, brown, natural, aged"
    ],
    "furniture": [
        "wood furniture texture, polished, elegant",
        "upholstery texture, soft, patterned",
        "metal furniture texture, industrial, modern",
        "glass furniture texture, transparent, modern",
        "plastic furniture texture, smooth, modern"
    ],
    "vehicles": [
        "vehicle paint texture, metallic, glossy",
        "vehicle interior texture, leather, luxury",
        "vehicle engine texture, metal, industrial",
        "vehicle tire texture, rubber, textured",
        "vehicle chrome texture, reflective, polished"
    ],
    "weapons": [
        "metal weapon texture, steel, polished",
        "wood weapon texture, natural, aged",
        "plastic weapon texture, modern, smooth",
        "leather weapon texture, brown, natural",
        "fabric weapon texture, soft, natural"
    ],
    "jewelry": [
        "gold texture, metallic, shiny, luxury",
        "silver texture, metallic, polished",
        "diamond texture, crystalline, reflective",
        "pearl texture, iridescent, smooth",
        "gemstone texture, colorful, crystalline"
    ],
    "books": [
        "book cover texture, leather, aged",
        "paper texture, white, smooth, clean",
        "parchment texture, aged, historical",
        "cardboard texture, brown, textured",
        "fabric book texture, soft, natural"
    ],
    "tools": [
        "metal tool texture, steel, industrial",
        "wood tool texture, natural, aged",
        "plastic tool texture, modern, smooth",
        "rubber tool texture, grip, textured",
        "fabric tool texture, soft, natural"
    ],
    "sports": [
        "ball texture, leather, natural",
        "racket texture, string, tension",
        "glove texture, leather, natural",
        "uniform texture, fabric, team colors",
        "equipment texture, metal, industrial"
    ],
    "musical": [
        "wood instrument texture, polished, elegant",
        "metal instrument texture, brass, shiny",
        "string texture, natural, tension",
        "drum texture, leather, natural",
        "plastic instrument texture, modern, smooth"
    ],
    "medical": [
        "medical equipment texture, metal, sterile",
        "bandage texture, fabric, soft",
        "plastic medical texture, smooth, clean",
        "rubber medical texture, flexible, natural",
        "glass medical texture, transparent, clean"
    ],
    "kitchen": [
        "kitchen counter texture, stone, smooth",
        "kitchen cabinet texture, wood, polished",
        "kitchen appliance texture, metal, modern",
        "kitchen utensil texture, metal, polished",
        "kitchen fabric texture, soft, natural"
    ],
    "bathroom": [
        "bathroom tile texture, ceramic, smooth",
        "bathroom fixture texture, metal, polished",
        "bathroom fabric texture, soft, natural",
        "bathroom glass texture, transparent, clean",
        "bathroom plastic texture, smooth, modern"
    ],
    "office": [
        "desk texture, wood, polished, professional",
        "chair texture, fabric, comfortable",
        "paper texture, white, smooth, clean",
        "plastic office texture, modern, smooth",
        "metal office texture, industrial, modern"
    ],
    "garden": [
        "garden soil texture, brown, natural",
        "garden stone texture, natural, rough",
        "garden wood texture, natural, aged",
        "garden metal texture, rusted, weathered",
        "garden fabric texture, soft, natural"
    ]
} 