from django.shortcuts import render
from .models import DigitPrediction
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import os
from django.conf import settings

# Load CNN model once at startup
MODEL_PATH = os.path.join(settings.BASE_DIR, 'model', 'mnist_cnn_model.h5')
model = load_model(MODEL_PATH)

def upload_and_predict(request):
    if request.method == "POST" and request.FILES.get('digit_image'):
        uploaded_file = request.FILES['digit_image']
        # Save uploaded image in model
        dp = DigitPrediction.objects.create(image=uploaded_file)

        # Preprocess image to match training preprocessing
        img = Image.open(dp.image.path).convert('L')  # grayscale
        img = img.resize((28,28))
        img_array = np.array(img).astype('float32') / 255.0

        # Check if image needs inversion (MNIST format: black digits on white background)
        # If image has mostly dark pixels (white digits on black), invert it
        mean_pixel_value = np.mean(img_array)
        if mean_pixel_value < 0.5:  # Image is mostly dark (white digits on black background)
            img_array = 1 - img_array  # Invert to match MNIST format

        img_array = img_array.reshape(1, 28, 28, 1)

        # Predict digit
        prediction = model.predict(img_array)
        dp.predicted_class = int(np.argmax(prediction))
        dp.save()

        return render(request, 'digit_recognition/result.html', {'prediction': dp.predicted_class})
    
    return render(request, 'digit_recognition/upload.html')
