CAR DETECTION APP (STREAMLIT + YOLO)
====================================

How it works
------------
  Train in Google Colab (your notebook)  ->  download best.pt
  app.py loads  models/best.pt           ->  detects cars in uploaded images

Folder layout
-------------
  car_detection/
    app.py
    requirements.txt
    classes.txt
    README.txt
    models/
      best.pt      <- put the file you downloaded from Colab here

Steps
-----
1. Install:
     pip install -r requirements.txt

2. Put best.pt inside the models/ folder.

3. Run the app:
     streamlit run app.py

Edit classes.txt
----------------
Your dataset's classes are named 0, 1, 2, 3. Class 0 is most likely "car".
Rename the lines in classes.txt in class order (line 1 = class 0).

Notes
-----
- If models/best.pt is missing, the app falls back to the generic yolo11n.pt
  model and shows a warning.
- Cars in your images are tiny (drone shots), so keep the inference image
  size at 640 or more.
