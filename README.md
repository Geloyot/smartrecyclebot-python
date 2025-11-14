# SMART RECYCLEBOT - Detection Service

YOLOv8-based waste classification detection service.

## Features
- 🤖 Real-time object detection
- 🎯 Waste classification (biodegradable/non-biodegradable)
- 📡 RESTful API for camera control
- 🔄 Integration with Laravel web app

## Local Setup

### Prerequisites
- Python 3.9+
- Webcam/Camera device

### Installation
```bash
# Clone repository
git clone <your-repo-url>
cd smartrecyclebot-python

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Environment setup
cp .env.example .env
```

### Running
```bash
# Start detection service
uvicorn main:app --host 0.0.0.0 --port 8001 --reload
```

## API Endpoints
- `POST /camera/start` - Start detection
- `POST /camera/stop` - Stop detection
- `GET /camera/status` - Check camera status

## Model
Uses YOLOv8 trained on custom waste classification dataset.
Model file: `best.pt`