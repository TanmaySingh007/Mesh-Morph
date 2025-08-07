#!/bin/bash

# Heroku Deployment Script for MeshMorph

echo "🚀 Deploying MeshMorph to Heroku..."

# Check if Heroku CLI is installed
if ! command -v heroku &> /dev/null; then
    echo "❌ Heroku CLI not found. Please install it first:"
    echo "   https://devcenter.heroku.com/articles/heroku-cli"
    exit 1
fi

# Login to Heroku
echo "🔐 Logging into Heroku..."
heroku login

# Create new Heroku app (if not exists)
echo "📦 Creating Heroku app..."
heroku create mesh-morph-ai --buildpack heroku/python

# Set environment variables
echo "⚙️ Setting environment variables..."
heroku config:set MODEL_CACHE_DIR=/app/.cache
heroku config:set OUTPUT_DIR=/app/generated_textures

# Deploy to Heroku
echo "🚀 Deploying to Heroku..."
git push heroku main

# Open the app
echo "🌐 Opening the deployed app..."
heroku open

echo "✅ Deployment completed!"
echo "🌐 Your app is now live at: https://mesh-morph-ai.herokuapp.com"
