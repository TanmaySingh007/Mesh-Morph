#!/bin/bash

# Google Cloud Run Deployment Script for MeshMorph

echo "🚀 Deploying MeshMorph to Google Cloud Run..."

# Set your project ID (replace with your actual project ID)
PROJECT_ID="your-project-id"

# Build and deploy to Cloud Run
gcloud run deploy mesh-morph-ai \
  --source . \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated \
  --memory 2Gi \
  --cpu 1 \
  --timeout 300 \
  --concurrency 1 \
  --max-instances 10 \
  --project $PROJECT_ID

echo "✅ Deployment completed!"
echo "🌐 Your app is now live at the URL shown above"
