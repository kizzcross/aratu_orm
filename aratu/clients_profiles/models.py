from django.db import models

# Here we will have the real user profile, so we can add more fields to the user

class Profile(models.Model):
    user = models.OneToOneField('auth.User', on_delete=models.CASCADE)
    