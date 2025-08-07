# 🎨 MeshMorph - AI-Powered Texture & 3D Asset Generator

> **Transform your creative ideas into stunning textures and 3D assets using cutting-edge AI technology**

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://python.org)
[![Flask](https://img.shields.io/badge/Flask-2.0+-red.svg)](https://flask.palletsprojects.com/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-orange.svg)](https://pytorch.org/)
[![Stable Diffusion](https://img.shields.io/badge/Stable%20Diffusion-v1.5-green.svg)](https://huggingface.co/runwayml/stable-diffusion-v1-5)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## 🌟 What is MeshMorph?

MeshMorph is an intelligent web application that bridges the gap between imagination and digital creation. It uses state-of-the-art AI models to generate high-quality textures and 3D assets from simple text descriptions. Whether you're a game developer, 3D artist, or creative professional, MeshMorph empowers you to bring your ideas to life instantly.

### ✨ Key Features

- **🎨 AI-Powered Texture Generation**: Create realistic textures from text prompts
- **🧊 3D Asset Creation**: Generate textured 3D models (cube, sphere, cylinder)
- **🌐 Beautiful Web Interface**: Intuitive, responsive design for seamless interaction
- **📥 Easy Export**: Download assets in standard formats (PNG, OBJ, MTL)
- **⚡ Smart Optimization**: Optimized for both CPU and GPU environments
- **🎯 Material Presets**: Quick access to common materials and objects

## 🏗️ Technology Stack & Architecture

### 🤖 **AI/ML Foundation**

#### **PyTorch 2.0+**
- **Why PyTorch?** Chosen for its dynamic computational graphs and excellent GPU support
- **Benefits**: Enables real-time model optimization and efficient memory management
- **Impact**: Faster inference times and better resource utilization

#### **Stable Diffusion v1.5**
- **Model**: `runwayml/stable-diffusion-v1-5`
- **Why Stable Diffusion?** Industry-leading text-to-image generation with exceptional quality
- **Capabilities**: 512x512 resolution, 20 inference steps, guidance scale 7.5
- **Optimizations**: Attention slicing, VAE slicing, memory-efficient attention

#### **Hugging Face Diffusers**
- **Purpose**: Provides the Stable Diffusion pipeline implementation
- **Benefits**: Easy model loading, caching, and optimization
- **Features**: Automatic model downloading and version management

### 🌐 **Web Framework**

#### **Flask 2.0+**
- **Why Flask?** Lightweight, flexible, and perfect for AI applications
- **Benefits**: 
  - Minimal overhead for AI model serving
  - Easy integration with Python ML libraries
  - Rapid development and deployment
- **Features**: RESTful API endpoints, file serving, error handling

#### **Gunicorn (Production)**
- **Purpose**: Production WSGI server for deployment
- **Configuration**: Single worker with 300-second timeout for AI processing
- **Benefits**: Stable, scalable, and optimized for long-running AI tasks

### 🎨 **Frontend Technologies**

#### **Bootstrap 5.3**
- **Why Bootstrap?** Responsive design framework with excellent mobile support
- **Features**: Grid system, components, utilities
- **Benefits**: Consistent UI across devices, rapid development

#### **Modern CSS3**
- **Features**: Glass morphism effects, gradients, animations
- **Benefits**: Beautiful, modern interface that enhances user experience
- **Implementation**: Custom CSS with responsive design principles

#### **JavaScript (ES6+)**
- **Purpose**: Dynamic interactions and AJAX requests
- **Features**: Async/await, fetch API, DOM manipulation
- **Benefits**: Smooth user experience with real-time updates

### 🧊 **3D Processing**

#### **Trimesh**
- **Purpose**: Python library for 3D mesh manipulation
- **Capabilities**: Mesh creation, UV mapping, file export
- **Benefits**: Generates standard 3D file formats (OBJ, MTL)

#### **NumPy & SciPy**
- **Purpose**: Numerical computing for 3D geometry
- **Benefits**: Efficient mathematical operations for mesh generation
- **Features**: Array operations, linear algebra, optimization

### 🎯 **Key Libraries & Dependencies**

```python
# Core AI Libraries
torch>=2.0.0          # Deep learning framework
diffusers>=0.21.0      # Stable Diffusion implementation
transformers>=4.30.0   # Model loading and inference
accelerate>=0.20.0     # Optimized model acceleration

# 3D & Image Processing
trimesh>=3.20.0        # 3D mesh creation and manipulation
Pillow>=9.5.0          # Image processing and format conversion
numpy>=1.24.0          # Numerical computing for 3D geometry
scipy>=1.10.0          # Scientific computing utilities

# Web Development
Flask>=2.3.0           # Web framework
gunicorn>=20.1.0       # Production WSGI server
```

## 🚀 How It Works

### 1. **Text-to-Texture Pipeline**
```
User Input → AI Model → Texture Generation → Enhancement → Download
```

1. **User enters a text description** (e.g., "rough stone wall texture")
2. **AI model processes the prompt** using Stable Diffusion
3. **Texture is generated** with specified parameters (resolution, steps)
4. **Post-processing enhancement** improves quality and detail
5. **File is saved and made available** for download

### 2. **3D Asset Creation**
```
Texture Generation → 3D Mesh Creation → UV Mapping → File Export
```

1. **Generate texture** using the same AI pipeline
2. **Create 3D mesh** (cube, sphere, cylinder) using Trimesh
3. **Apply texture mapping** with proper UV coordinates
4. **Export files** (OBJ, MTL, PNG) for 3D software compatibility

### 3. **Web Interface Flow**
```
Browser → Flask API → AI Model → Response → Frontend Display
```

- **Responsive design** adapts to any device
- **Real-time status updates** keep users informed
- **Progress indication** shows generation progress
- **Error handling** provides clear feedback

## 🎯 **Why This Technology Stack?**

### **AI/ML Choices**
- **Stable Diffusion**: Best-in-class text-to-image generation
- **PyTorch**: Industry standard with excellent GPU support
- **Hugging Face**: Reliable model hosting and versioning

### **Web Framework Benefits**
- **Flask**: Lightweight, perfect for AI applications
- **Gunicorn**: Production-ready with long timeout support
- **Simple deployment**: Easy to containerize and scale

### **Frontend Advantages**
- **Bootstrap**: Rapid development with mobile-first design
- **Modern CSS**: Beautiful, engaging user interface
- **JavaScript**: Smooth, responsive interactions

### **3D Processing**
- **Trimesh**: Python-native 3D library
- **Standard formats**: OBJ/MTL compatibility with all 3D software
- **Efficient**: Fast mesh generation and texture mapping

## 🛠️ Installation & Setup

### **Prerequisites**
- Python 3.9 or higher
- 8GB+ RAM (16GB recommended)
- 5GB+ free disk space for model downloads
- Modern web browser with JavaScript enabled

### **Quick Start**

1. **Clone the repository**
   ```bash
   git clone https://github.com/TanmaySingh007/Mesh-Morph.git
   cd Mesh-Morph
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application**
   ```bash
   python web_app.py
   ```

4. **Open your browser**
   Navigate to `http://localhost:5000`

### **First Run**
- The AI model (~4GB) will download automatically on first use
- Generation may take 2-5 minutes on CPU
- GPU acceleration significantly improves performance

## 🎮 Usage Guide

### **2D Texture Generation**

1. **Navigate to "2D Texture Generation" tab**
2. **Enter a description** (e.g., "detailed oak wood grain texture")
3. **Adjust settings**:
   - **Quality Steps**: 10 (fast) to 30 (ultra quality)
   - **Dimensions**: 256x256 to 1024x1024
   - **Enhance**: Enable for better quality
4. **Click "Generate Texture"**
5. **Download the PNG file**

### **3D Asset Generation**

1. **Navigate to "3D Asset Generation" tab**
2. **Enter a texture description** (e.g., "stone wall texture")
3. **Choose 3D shape**:
   - **Cube**: 6-sided textured cube
   - **Sphere**: Textured sphere with UV mapping
   - **Cylinder**: Textured cylinder with proper mapping
4. **Adjust generation parameters**
5. **Click "Generate 3D Asset"**
6. **Download all three files**:
   - **OBJ**: 3D model file
   - **MTL**: Material definition file
   - **PNG**: Texture image file

### **Using Material Presets**

Click on any preset button for instant material generation:
- **Natural**: Wood, Stone, Metal, Fabric, Leather
- **Synthetic**: Plastic, Glass, Ceramic, Concrete
- **Objects**: Cars, Electronics, Furniture, Clothing
- **Specialized**: Medical, Sports, Musical, Jewelry

## 🚀 Deployment Options

### **Render (Recommended)**

**Why Render?**
- **GPU Support**: Better performance for AI model inference
- **Longer Timeouts**: Up to 30 minutes vs Vercel's 10-second limit
- **Resource Scaling**: Can handle heavy computational loads
- **Cost Effective**: Pay-per-use pricing model

**Deployment Steps:**
1. Fork/clone this repository to your GitHub account
2. Connect to [Render.com](https://render.com)
3. Create new Web Service
4. Configure with provided `render.yaml`
5. Deploy with one click

### **Local Development**
```bash
python web_app.py
```

### **Production Deployment**
```bash
gunicorn web_app:app --bind 0.0.0.0:8000 --workers 1 --timeout 300
```

### **Docker Deployment**
```bash
docker build -t mesh-morph-ai .
docker run -p 8000:8000 mesh-morph-ai
```

## 🔧 **API Endpoints**

### **Core Endpoints**
```
GET  /                    # Main application interface
GET  /status              # System status and readiness
GET  /health              # Health check endpoint
POST /generate_texture    # Generate 2D texture
POST /generate_3d_asset   # Generate 3D asset
GET  /download/<file>     # Download generated files
```

### **Example API Usage**

**Generate 2D Texture:**
```bash
curl -X POST http://localhost:5000/generate_texture \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "detailed wood grain texture",
    "width": 512,
    "height": 512,
    "steps": 15,
    "enhance": true
  }'
```

**Generate 3D Asset:**
```bash
curl -X POST http://localhost:5000/generate_3d_asset \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "stone wall texture",
    "shape": "cube",
    "width": 512,
    "height": 512,
    "steps": 15,
    "enhance": true
  }'
```

## 🏗️ **Project Structure**

```
Mesh-Morph/
├── 📁 templates/              # HTML templates
│   └── index.html            # Main application interface
├── 📁 generated_textures/     # Output directory for generated files
├── 📄 web_app.py             # Flask web application
├── 📄 main.py                # Core AI texture generator
├── 📄 config.py              # Configuration settings
├── 📄 utils.py               # Utility functions
├── 📄 utils_3d.py           # 3D mesh generation utilities
├── 📄 requirements.txt       # Python dependencies
├── 📄 render.yaml           # Render deployment configuration
├── 📄 Dockerfile            # Docker container setup
├── 📄 README.md             # This file
└── 📄 vercel.json           # Vercel deployment configuration
```

## 🔧 **Performance Optimization**

### **Model Optimization**
- **Attention Slicing**: Reduces memory usage by 30-50%
- **VAE Slicing**: Optimizes image processing pipeline
- **Memory Efficient Attention**: Better GPU utilization
- **Float16 Precision**: Faster inference (use float32 for CPU)

### **System Requirements**
- **Minimum**: 8GB RAM, 4-core CPU
- **Recommended**: 16GB RAM, 8-core CPU, GPU
- **Storage**: 5GB for models + generated files
- **Network**: Stable internet for model downloads

### **Performance Tips**
- **Use GPU** for faster generation (CUDA required)
- **Reduce inference steps** for faster results
- **Use smaller dimensions** for quick previews
- **Enable model optimizations** in config

## 🐛 **Troubleshooting**

### **Common Issues**

**Model Download Fails:**
```bash
# Clear Hugging Face cache
rm -rf ~/.cache/huggingface/
# Reinstall with specific version
pip install diffusers==0.21.0
```

**Out of Memory:**
```python
# In config.py, use CPU mode
DEFAULT_MODEL_CONFIG = ModelConfig(
    torch_dtype="float32",  # Use float32 for CPU
    device="cpu"
)
```

**Download Issues:**
- Ensure `generated_textures/` directory exists
- Check file permissions
- Verify Flask has write access

### **Performance Tips**
- Use GPU for faster generation (CUDA required)
- Reduce inference steps for faster results
- Use smaller dimensions for quick previews
- Enable model optimizations in config

## 🤝 **Contributing**

We welcome contributions! Here's how you can help:

### **Development Setup**
```bash
git clone https://github.com/TanmaySingh007/Mesh-Morph.git
cd Mesh-Morph
pip install -r requirements.txt
python web_app.py
```

### **Code Style**
- Follow PEP 8 Python style guide
- Use type hints for function parameters
- Add docstrings for all functions
- Write unit tests for new features

### **Contributing Guidelines**
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 **License**

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 **Acknowledgments**

- **Hugging Face**: For the amazing Diffusers library
- **Stability AI**: For the Stable Diffusion model
- **PyTorch Team**: For the excellent deep learning framework
- **Flask Community**: For the lightweight web framework
- **Bootstrap Team**: For the responsive UI framework
- **Trimesh Developers**: For the 3D mesh processing library

## 📞 **Support**

- **Issues**: [GitHub Issues](https://github.com/TanmaySingh007/Mesh-Morph/issues)
- **Discussions**: [GitHub Discussions](https://github.com/TanmaySingh007/Mesh-Morph/discussions)
- **Documentation**: Check the [Wiki](https://github.com/TanmaySingh007/Mesh-Morph/wiki)

---

**Made with ❤️ by the MeshMorph Team**

*Transform your ideas into stunning 3D textures and assets with the power of AI!* 🎨✨ 