# Safe Distance App using MoveNet TF model and OpenCV on Flask

There are two files: KivyApp, which uses Kivy and can only detect one person, and Flask app, which uses MoveNet, a multiple detection model that is considered the main app.

# Instructions

1. `python -m venv .env`
2. `.\.env\Scripts\activate`
3. `pip install -r .\requirements.txt`
4. `python .\app.py`

If you are not using the default device camera, you may need to change the `cameraIndex` value in the app.py file. If setting the value to 1 does not work, try increasing the value until the intended camera is found.

In app.py file, line 22
```
"""
👋👋👋👋
please change me, if the camera did not work
"""
cameraIndex = 0
camera = cv2.VideoCapture(cameraIndex)
```