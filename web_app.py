"""
Minimal Web Application for Mesh Morph - Ultra Fast Vercel Deployment
"""

from flask import Flask, render_template, request, jsonify, send_file
import json
from datetime import datetime
import os
import io
from PIL import Image
from main import TextureGenerator
from config import DEFAULT_MODEL_CONFIG, DEFAULT_GENERATION_CONFIG, DEFAULT_OUTPUT_CONFIG

app = Flask(__name__)

# Initialize the texture generator
generator = None

def initialize_generator():
    """Initialize the texture generator."""
    global generator
    try:
        generator = TextureGenerator()
        generator.load_model()
        return True
    except Exception as e:
        print(f"Error initializing generator: {e}")
        return False

@app.route('/')
def index():
    """Main page."""
    return render_template('index.html')

@app.route('/status')
def status():
    """Check system status."""
    global generator
    if generator is None:
        # Try to initialize
        if initialize_generator():
            status_msg = "System Ready"
            status_type = "ready"
        else:
            status_msg = "Initializing..."
            status_type = "loading"
    else:
        status_msg = "System Ready"
        status_type = "ready"
    
    return jsonify({
        'status': status_type, 
        'mode': 'production',
        'message': status_msg,
        'timestamp': datetime.now().isoformat()
    })

@app.route('/health')
def health():
    """Health check endpoint."""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'version': '1.0.0-production'
    })

@app.route('/generate_texture', methods=['POST'])
def generate_texture():
    """Generate 2D texture from text prompt."""
    try:
        global generator
        if generator is None:
            if not initialize_generator():
                return jsonify({'error': 'Texture generator not available'}), 500
        
        data = request.get_json()
        prompt = data.get('prompt', 'wood texture')
        width = data.get('width', 512)
        height = data.get('height', 512)
        steps = data.get('steps', 20)
        enhance = data.get('enhance', True)
        
        # Limit steps for faster generation on CPU
        if steps > 20:
            steps = 20
        
        # Generate the texture
        texture = generator.generate_texture_direct(
            prompt=prompt,
            width=width,
            height=height,
            num_inference_steps=steps,
            enhance=enhance
        )
        
        # Save the texture
        filename = f"texture_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        output_path = os.path.join(DEFAULT_OUTPUT_CONFIG.output_dir, filename)
        os.makedirs(DEFAULT_OUTPUT_CONFIG.output_dir, exist_ok=True)
        texture.save(output_path)
        
        # Convert to base64 for immediate display
        img_io = io.BytesIO()
        texture.save(img_io, 'PNG')
        img_io.seek(0)
        
        return jsonify({
            'success': True,
            'message': 'Texture generated successfully',
            'prompt': prompt,
            'dimensions': f"{width}x{height}",
            'filename': filename,
            'download_url': f'/download/{filename}',
            'image_data': img_io.getvalue().hex()  # Send image data for immediate display
        })
        
    except Exception as e:
        print(f"Error generating texture: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/generate_3d_asset', methods=['POST'])
def generate_3d_asset():
    """Generate 3D asset from text prompt."""
    try:
        global generator
        if generator is None:
            if not initialize_generator():
                return jsonify({'error': 'Texture generator not available'}), 500
        
        data = request.get_json()
        prompt = data.get('prompt', 'stone texture')
        shape = data.get('shape', 'cube')
        width = data.get('width', 512)
        height = data.get('height', 512)
        steps = data.get('steps', 20)
        enhance = data.get('enhance', True)
        
        # Limit steps for faster generation on CPU
        if steps > 20:
            steps = 20
        
        # Generate the 3D asset
        obj_path = generator.generate_3d_asset(
            prompt=prompt,
            shape=shape,
            width=width,
            height=height,
            num_inference_steps=steps,
            enhance=enhance
        )
        
        # Get the actual filenames
        base_path = obj_path[:-4]  # remove .obj
        obj_filename = os.path.basename(obj_path)
        mtl_filename = os.path.basename(base_path + '.mtl')
        png_filename = os.path.basename(base_path + '.png')
        
        # Get the texture image for preview
        texture_path = base_path + '.png'
        if os.path.exists(texture_path):
            with open(texture_path, 'rb') as f:
                texture_data = f.read()
            image_data = texture_data.hex()
        else:
            image_data = None
        
        return jsonify({
            'success': True,
            'message': '3D asset generated successfully',
            'prompt': prompt,
            'shape': shape,
            'obj_filename': obj_filename,
            'mtl_filename': mtl_filename,
            'png_filename': png_filename,
            'download_obj_url': f'/download/{obj_filename}',
            'download_mtl_url': f'/download/{mtl_filename}',
            'download_png_url': f'/download/{png_filename}',
            'image_data': image_data  # Send texture image for preview
        })
        
    except Exception as e:
        print(f"Error generating 3D asset: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/download/<filename>')
def download_file(filename):
    """Download generated files."""
    try:
        file_path = os.path.join(DEFAULT_OUTPUT_CONFIG.output_dir, filename)
        if os.path.exists(file_path):
            return send_file(file_path, as_attachment=True)
        else:
            return jsonify({'error': 'File not found'}), 404
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/download_texture/<filename>')
def download_texture(filename):
    """Download texture file for 3D assets."""
    try:
        # Convert .obj filename to .png filename
        texture_filename = filename.replace('.obj', '.png')
        file_path = os.path.join(DEFAULT_OUTPUT_CONFIG.output_dir, texture_filename)
        if os.path.exists(file_path):
            return send_file(file_path, as_attachment=True)
        else:
            return jsonify({'error': 'Texture file not found'}), 404
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # Initialize the generator on startup
    initialize_generator()
    port = int(os.environ.get('PORT', 8080))
    app.run(debug=False, host='0.0.0.0', port=port) 