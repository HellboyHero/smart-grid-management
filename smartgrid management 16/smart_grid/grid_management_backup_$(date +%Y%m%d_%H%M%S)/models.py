from django.db import models
from django.contrib.auth.models import User
import uuid

class UserProfile(models.Model):
    user = models.OneToOneField(User, on_delete=models.CASCADE)
    customer_id = models.CharField(max_length=10, unique=True, null=True, blank=True)
    full_name = models.CharField(max_length=100, default='')
    door_number = models.CharField(max_length=50)
    area_name = models.CharField(max_length=100)
    location = models.TextField()
    service_no = models.CharField(max_length=13, unique=True)
    
    def save(self, *args, **kwargs):
        if not self.customer_id:
            # Generate customer ID: CUS followed by 7 random digits
            while True:
                new_id = 'CUS' + str(uuid.uuid4().int)[:7]
                if not UserProfile.objects.filter(customer_id=new_id).exists():
                    self.customer_id = new_id
                    break
        super().save(*args, **kwargs)
    
    def __str__(self):
        return f"{self.full_name}'s Profile" if self.full_name else f"{self.user.username}'s Profile"

class Bill(models.Model):
    user_profile = models.ForeignKey(UserProfile, on_delete=models.CASCADE)
    month = models.CharField(max_length=20)
    year = models.CharField(max_length=4)
    units_consumed = models.FloatField()
    rate_per_unit = models.FloatField()
    amount = models.FloatField()
    tax = models.FloatField()
    total_amount = models.FloatField()
    generated_date = models.DateTimeField(auto_now_add=True)
    
    def __str__(self):
        return f"Bill for {self.user_profile.full_name} - {self.month} {self.year}"

class PowerConsumption(models.Model):
    user_profile = models.ForeignKey(UserProfile, on_delete=models.CASCADE)
    timestamp = models.DateTimeField()
    actual_consumption = models.FloatField()  # in kWh
    predicted_consumption = models.FloatField()  # in kWh
    temperature = models.FloatField()  # in Celsius
    solar_radiation = models.FloatField()  # in W/m²
    wind_speed = models.FloatField()  # in m/s
    humidity = models.FloatField()  # in %
    cloud_cover = models.FloatField()  # in %
    theft_detected = models.BooleanField(default=False)
    theft_probability = models.FloatField(default=0.0)
    
    class Meta:
        ordering = ['-timestamp']
        indexes = [
            models.Index(fields=['user_profile', 'timestamp']),
            models.Index(fields=['theft_detected']),
        ]
    
    def __str__(self):
        return f"Power Consumption - {self.user_profile.full_name} at {self.timestamp}"

class LoadBalancingMetrics(models.Model):
    timestamp = models.DateTimeField(auto_now_add=True)
    average_load = models.FloatField()  # in MW
    peak_load = models.FloatField()  # in MW
    load_factor = models.FloatField()  # ratio
    peak_reduction = models.FloatField()  # percentage
    total_demand = models.FloatField()  # in MW
    balanced_demand = models.FloatField()  # in MW
    
    class Meta:
        ordering = ['-timestamp']
        indexes = [
            models.Index(fields=['timestamp']),
        ]
    
    def __str__(self):
        return f"Load Balancing Metrics at {self.timestamp}"
