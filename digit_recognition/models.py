from django.db import models

# Create your models here.
from django.db import models

class DigitPrediction(models.Model):
    image = models.ImageField(upload_to='digit_images/')
    predicted_class = models.IntegerField(null=True)
    uploaded_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"Prediction: {self.predicted_class}"
