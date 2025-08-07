# MeshMorph - AI Texture Generation

An AI-powered web application for generating high-quality textures and 3D assets using Stable Diffusion.

## Features

- 🎨 **AI Texture Generation**: Generate textures from text prompts using Stable Diffusion
- 🎯 **3D Asset Creation**: Create textured 3D models (cube, sphere, cylinder)
- 📱 **Web Interface**: User-friendly web UI for easy interaction
- 🚀 **Fast Generation**: Optimized for quick texture generation
- 💾 **Download Support**: Download generated textures and 3D assets

## Local Development

### Prerequisites
- Python 3.9+
- pip

### Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd MeshMorph-master
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the application:
```bash
python web_app.py
```

4. Open your browser and go to `http://localhost:5000`

## Deployment on Render

This application is optimized for deployment on Render due to its AI model requirements.

### Why Render?

- **GPU Support**: Better performance for AI model inference
- **Longer Timeouts**: Up to 30 minutes vs Vercel's 10-second limit
- **Resource Scaling**: Can handle heavy computational loads
- **Cost Effective**: Pay-per-use pricing model

### Deployment Steps

1. **Fork/Clone** this repository to your GitHub account

2. **Connect to Render**:
   - Go to [render.com](https://render.com)
   - Sign up/Login with your GitHub account
   - Click "New +" → "Web Service"

3. **Configure the Service**:
   - **Name**: `mesh-morph-ai`
   - **Environment**: `Python`
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `gunicorn web_app:app --bind 0.0.0.0:$PORT --workers 1 --timeout 300`

4. **Environment Variables** (optional):
   - `MODEL_CACHE_DIR`: `/opt/render/project/src/.cache`
   - `OUTPUT_DIR`: `/opt/render/project/src/generated_textures`

5. **Deploy**: Click "Create Web Service"

### Alternative: Use render.yaml

The repository includes a `render.yaml` file for easy deployment:

1. Connect your GitHub repository to Render
2. Render will automatically detect the `render.yaml` file
3. Deploy with one click

## Usage

### Web Interface

1. **Generate 2D Texture**:
   - Enter a text prompt describing the texture
   - Adjust parameters (width, height, steps)
   - Click "Generate Texture"
   - Download the result

2. **Generate 3D Asset**:
   - Enter a text prompt
   - Select shape (cube, sphere, cylinder)
   - Click "Generate 3D Asset"
   - Download the OBJ file and texture

### API Endpoints

- `GET /` - Main web interface
- `GET /status` - System status
- `GET /health` - Health check
- `POST /generate_texture` - Generate 2D texture
- `POST /generate_3d_asset` - Generate 3D asset
- `GET /download/<filename>` - Download files

## Configuration

The application uses several configuration files:

- `config.py` - Model and generation settings
- `utils.py` - Utility functions for texture processing
- `utils_3d.py` - 3D asset generation utilities

## Performance Tips

- **Local Development**: Use CPU for development (slower but no GPU required)
- **Production**: Consider GPU instances on Render for faster generation
- **Caching**: Models are cached locally to speed up subsequent generations
- **Batch Processing**: Generate multiple textures efficiently

## License

This project is licensed under the MIT License.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## Support

For issues and questions, please open an issue on GitHub. 