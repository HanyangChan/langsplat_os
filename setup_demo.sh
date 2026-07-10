#!/bin/bash
# Setup script for LangSplat Quick Demo

echo "Installing required Python packages for the demo..."
pip install gradio torch torchvision numpy matplotlib pillow
pip install git+https://github.com/openai/CLIP.git

echo ""
echo "Installation complete!"
echo "To run the demo, execute: python demo_app.py"
echo ""
echo "Note: The demo currently uses dummy data if you haven't downloaded the pretrained LangSplat models."
echo "To use real data, please download the pretrained model from the official LangSplat repository (Google Drive / Baidu) and place the rendered 3D features and decoder.pth in the 'sample_data' folder."
