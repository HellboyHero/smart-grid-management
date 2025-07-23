import os
import pandas as pd
import numpy as np
import joblib
from django.shortcuts import render
from django.http import JsonResponse
from django.conf import settings
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.datasets import make_regression
import random
from datetime import datetime
import pytz
import math
import json
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from django.contrib.auth.decorators import login_required
from .models import UserProfile, Bill, PowerConsumption, LoadBalancingMetrics

def previous_bills(request):
    return render(request, 'grid_management/previous_bills.html')

def load_energy_demand_data():
    try:
        energy_demand_df = pd.read_csv(os.path.join(settings.BASE_DIR.parent, "dataset/spg.csv"))
        # Rename the truncated column if needed
        if 'generated_powe' in energy_demand_df.columns:
            energy_demand_df = energy_demand_df.rename(columns={'generated_powe': 'generated_power_kw'})
        
        # Scale down the power values to simpler numbers (divide by 100)
        energy_demand_df['generated_power_kw'] = energy_demand_df['generated_power_kw'] / 100
        
        # Forward fill missing values for time series consistency
        energy_demand_df['generated_power_kw'] = energy_demand_df['generated_power_kw'].ffill()
        
        # Add an hour column if it doesn't exist
        if 'Hour' not in energy_demand_df.columns:
            energy_demand_df['Hour'] = range(len(energy_demand_df))
        return energy_demand_df
    except Exception as e:
        print(f"Error loading data: {str(e)}")
        return None

def load_model():
    try:
        # Get the absolute path to the models directory
        current_file = os.path.abspath(__file__)
        current_dir = os.path.dirname(current_file)
        model_dir = os.path.join(current_dir, 'models')
        
        # Create models directory if it doesn't exist
        os.makedirs(model_dir, exist_ok=True)
        
        model_path = os.path.join(model_dir, 'power_prediction_model.joblib')
        scaler_path = os.path.join(model_dir, 'scaler.joblib')
        
        # Check if model files exist
        if not os.path.exists(model_path) or not os.path.exists(scaler_path):
            print("Training new model...")
            model, scaler = train_new_model()
            
            # Save the model and scaler
            joblib.dump(model, model_path)
            joblib.dump(scaler, scaler_path)
            print("Model saved successfully")
            return model, scaler
        
        # Load existing model
        print("Loading existing model...")
        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
        return model, scaler
        
    except Exception as e:
        print(f"Error in load_model: {str(e)}")
        return None, None

def train_new_model():
    try:
        # Generate synthetic training data
        n_samples = 1000
        np.random.seed(42)
        
        # Create feature matrix with optimal ranges
        X = np.random.rand(n_samples, 5)
        
        # Temperature: 15-35°C (optimal: 25-30°C)
        X[:, 0] = X[:, 0] * 20 + 15
        
        # Solar radiation: 200-1000 W/m² (higher is better)
        X[:, 1] = X[:, 1] * 800 + 200
        
        # Wind speed: 0-15 m/s (optimal: 4-6 m/s)
        X[:, 2] = X[:, 2] * 15
        
        # Humidity: 30-90% (lower is better)
        X[:, 3] = X[:, 3] * 60 + 30
        
        # Cloud cover: 0-100% (lower is better)
        X[:, 4] = X[:, 4] * 100
        
        # Generate target values (power generation in MW) with realistic relationships
        y = np.zeros(n_samples)
        
        for i in range(n_samples):
            # Base load
            power = 1000
            
            # Temperature effect (optimal around 25-30°C)
            temp_effect = -abs(X[i, 0] - 27.5) * 10  # Penalize deviation from optimal temp
            
            # Solar radiation effect (linear positive relationship)
            solar_effect = X[i, 1] * 0.8  # More sun = more power
            
            # Wind speed effect (optimal around 4-6 m/s)
            wind_effect = 100 * (1 - abs(X[i, 2] - 5) / 5)  # Peak at 5 m/s
            
            # Humidity effect (negative relationship)
            humidity_effect = -X[i, 3] * 2  # Higher humidity = less power
            
            # Cloud cover effect (negative relationship)
            cloud_effect = -X[i, 4] * 3  # More clouds = less power
            
            # Combine all effects
            y[i] = power + temp_effect + solar_effect + wind_effect + humidity_effect + cloud_effect
        
        # Add some random noise
        y += np.random.normal(0, 50, n_samples)
        
        # Ensure no negative values
        y = np.maximum(y, 0)
        
        # Create and fit scaler
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Create and train model
        model = GradientBoostingRegressor(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=3,
            random_state=42
        )
        model.fit(X_scaled, y)
        
        # Test prediction with optimal values
        test_input = np.array([[
            27.5,  # Optimal temperature
            800,   # Good solar radiation
            5.0,   # Optimal wind speed
            45,    # Low humidity
            20     # Low cloud cover
        ]])
        test_scaled = scaler.transform(test_input)
        test_pred = model.predict(test_scaled)
        print(f"Test prediction with optimal conditions: {test_pred[0]:.2f} MW")
        
        return model, scaler
        
    except Exception as e:
        print(f"Error in train_new_model: {str(e)}")
        return None, None

def detect_power_theft(actual_demand, predicted_demand):
    """
    Detect potential power theft by comparing actual vs predicted demand
    Returns: (theft_detected, difference_mw, message)
    """
    threshold = 0.25  # 25% difference threshold
    difference = abs(actual_demand - predicted_demand)
    percentage_diff = (difference / predicted_demand) * 100 if predicted_demand != 0 else 0
    
    if percentage_diff > threshold * 100:
        if actual_demand > predicted_demand:
            message = (f" Potential power theft detected:\n"
                      f"Actual: {actual_demand:.2f} MW\n"
                      f"Expected: {predicted_demand:.2f} MW\n"
                      f"Excess: {difference:.2f} MW ({percentage_diff:.1f}% higher)")
            return True, difference, message
        else:
            message = (f" Unusual pattern detected:\n"
                      f"Actual: {actual_demand:.2f} MW\n"
                      f"Expected: {predicted_demand:.2f} MW\n"
                      f"Deficit: {difference:.2f} MW ({percentage_diff:.1f}% lower)")
            return True, difference, message
    else:
        message = (f" No power theft detected\n"
                  f"Actual: {actual_demand:.2f} MW\n"
                  f"Expected: {predicted_demand:.2f} MW\n"
                  f"Difference: {difference:.2f} MW ({percentage_diff:.1f}%)")
        return False, difference, message

def load_balancing(energy_demand_df, demand_adjustment):
    if 'generated_power_kw' not in energy_demand_df.columns and 'generated_powe' in energy_demand_df.columns:
        energy_demand_df = energy_demand_df.rename(columns={'generated_powe': 'generated_power_kw'})
    
    original_demand = energy_demand_df['generated_power_kw'].ffill()
    adjusted_demand = original_demand * (1 + demand_adjustment / 100)
    
    # Calculate load balancing metrics
    average_load = float(adjusted_demand.mean())
    peak_reduction = float(
        ((original_demand.max() - adjusted_demand.max()) / original_demand.max()) * 100
        if original_demand.max() > 0 else 0
    )
    
    energy_demand_df['Adjusted Demand (MW)'] = adjusted_demand
    return energy_demand_df, average_load, peak_reduction

@login_required(login_url='grid_management:login')
def index(request):
    energy_demand_df = load_energy_demand_data()
    if energy_demand_df is None:
        return render(request, 'grid_management/index.html', {'error': 'Failed to load energy demand data'})
    
    # List of features for the prediction form
    features = [
        'temperature_2_m_above_gnd', 'relative_humidity_2_m_above_gnd',
        'mean_sea_level_pressure_MSL', 'total_precipitation_sfc',
        'snowfall_amount_sfc', 'total_cloud_cover_sfc',
        'high_cloud_cover_high_cld_lay', 'medium_cloud_cover_mid_cld_lay',
        'low_cloud_cover_low_cld_lay', 'shortwave_radiation_backwards_sfc',
        'wind_speed_10_m_above_gnd', 'wind_direction_10_m_above_gnd',
        'wind_speed_80_m_above_gnd', 'wind_direction_80_m_above_gnd',
        'wind_speed_900_mb', 'wind_direction_900_mb',
        'wind_gust_10_m_above_gnd', 'angle_of_incidence', 'zenith', 'azimuth'
    ]
    
    # Convert DataFrame to dict for template rendering
    energy_data = {
        'hours': energy_demand_df['Hour'].tolist(),
        'demand': energy_demand_df['generated_power_kw'].ffill().tolist(),
    }
    
    return render(request, 'grid_management/index.html', {
        'energy_data': energy_data,
        'features': features,
    })

@login_required(login_url='grid_management:login')
def predict(request):
    if request.method == 'POST':
        try:
            # Load the model and scaler
            model, scaler = load_model()
            if model is None or scaler is None:
                return JsonResponse({'error': 'Failed to load prediction model'})
            
            # Get current real-time demand
            real_time_data = json.loads(get_real_time_data(request).content)
            current_demand_gw = real_time_data.get('current_demand', 0)
            current_demand_mw = current_demand_gw * 1000  # Convert GW to MW
            
            # Get input values from POST request
            temperature = float(request.POST.get('temperature', 25))
            solar_radiation = float(request.POST.get('solar_radiation', 500))
            wind_speed = float(request.POST.get('wind_speed', 5))
            humidity = float(request.POST.get('humidity', 60))
            cloud_cover = float(request.POST.get('cloud_cover', 50))
            
            # Create input array
            X = np.array([[
                temperature,
                solar_radiation,
                wind_speed,
                humidity,
                cloud_cover
            ]])
            
            # Scale the input
            X_scaled = scaler.transform(X)
            
            # Make prediction
            prediction = model.predict(X_scaled)[0]
            
            # Round to 2 decimal places
            prediction = round(prediction, 2)
            
            # Detect power theft
            theft_detected, difference, theft_message = detect_power_theft(current_demand_mw, prediction)
            
            return JsonResponse({
                'prediction': prediction,
                'unit': 'MW',
                'current_demand_gw': round(current_demand_gw, 2),
                'actual_demand': round(current_demand_mw, 2),
                'theft_detected': theft_detected,
                'theft_difference': round(difference, 2),
                'theft_message': theft_message
            })
            
        except Exception as e:
            print(f"Error in predict: {str(e)}")
            return JsonResponse({'error': str(e)})
    
    return JsonResponse({'error': 'Invalid request method'})

def calculate_theft_probability(actual_demand, predicted_demand, temperature, solar_radiation):
    """
    Calculate probability of power theft based on multiple factors
    Returns probability between 0 and 1
    """
    # Base probability from demand difference
    difference_mw = abs(actual_demand - predicted_demand)
    base_prob = min(difference_mw / 10.0, 1.0)  # Scale difference to probability
    
    # Adjust based on temperature (higher probability during extreme temperatures)
    temp_factor = abs(temperature - 22) / 20  # 22°C is considered optimal
    
    # Adjust based on solar radiation (higher probability during high generation periods)
    solar_factor = solar_radiation / 1000  # Normalize to 0-1 range
    
    # Combine factors with weights
    final_prob = (0.6 * base_prob + 0.2 * temp_factor + 0.2 * solar_factor)
    
    return min(max(final_prob, 0.0), 1.0)  # Ensure probability is between 0 and 1

def update_load_balancing_metrics(actual_demand, predicted_demand):
    """
    Update load balancing metrics based on current demand values
    """
    # Get recent metrics for trend analysis
    recent_metrics = LoadBalancingMetrics.objects.order_by('-timestamp')[:24]  # Last 24 records
    
    if recent_metrics:
        avg_load = sum(m.average_load for m in recent_metrics) / len(recent_metrics)
        peak_load = max(m.peak_load for m in recent_metrics)
    else:
        avg_load = actual_demand
        peak_load = actual_demand
    
    # Calculate load factor (ratio of average to peak load)
    load_factor = avg_load / peak_load if peak_load > 0 else 1.0
    
    # Calculate peak reduction (if any)
    peak_reduction = ((peak_load - actual_demand) / peak_load * 100) if actual_demand < peak_load else 0.0
    
    # Calculate balanced demand (smoothed demand)
    balanced_demand = (actual_demand + predicted_demand) / 2
    
    LoadBalancingMetrics.objects.create(
        average_load=avg_load,
        peak_load=peak_load,
        load_factor=load_factor,
        peak_reduction=peak_reduction,
        total_demand=actual_demand,
        balanced_demand=balanced_demand
    )

@login_required(login_url='grid_management:login')
def adjust_demand(request):
    if request.method == 'POST':
        try:
            adjustment = float(request.POST.get('adjustment', 0))
            transmission_distance = float(request.POST.get('distance', 100))
            
            # Get current hour's demand
            ist = pytz.timezone('Asia/Kolkata')
            current_hour = datetime.now(ist).hour
            
            # Get base load for current hour
            base_loads = {
                0: 800, 1: 750, 2: 700, 3: 680, 4: 690, 5: 720,
                6: 850, 7: 1000, 8: 1200, 9: 1300, 10: 1400, 11: 1450,
                12: 1400, 13: 1350, 14: 1300, 15: 1280, 16: 1300, 17: 1400,
                18: 1500, 19: 1600, 20: 1550, 21: 1400, 22: 1200, 23: 1000
            }
            
            # Generate original demand pattern
            original_demand = []
            for minute in range(60):
                base = base_loads[current_hour]
                minute_var = np.sin(minute * 6) * 50
                random_var = np.random.normal(0, 20)
                temp_var = np.sin(current_hour * 15 + minute/60) * 30
                demand = base + minute_var + random_var + temp_var
                original_demand.append(float(demand))
            
            # Calculate balanced demand with adjustment
            adjustment_factor = 1 + (adjustment / 100)
            balanced_demand = [d * adjustment_factor for d in original_demand]
            
            # Calculate metrics
            avg_original = np.mean(original_demand)
            peak_original = np.max(original_demand)
            avg_balanced = np.mean(balanced_demand)
            peak_balanced = np.max(balanced_demand)
            
            peak_reduction = ((peak_original - peak_balanced) / peak_original) * 100
            load_factor = (avg_balanced / peak_balanced) * 100
            
            # Generate distribution data
            distribution_data = generate_distribution_data(balanced_demand)
            
            # Add transmission distance to distribution data
            distribution_data['transmission_distance'] = transmission_distance
            
            # Convert numpy arrays to lists for JSON serialization
            distribution_data['input_power'] = [float(x) for x in distribution_data['input_power']]
            distribution_data['output_power'] = [float(x) for x in distribution_data['output_power']]
            distribution_data['power_loss'] = [float(x) for x in distribution_data['power_loss']]
            distribution_data['total_loss'] = float(distribution_data['total_loss'])
            
            response_data = {
                'original_demand': [float(x) for x in original_demand],
                'balanced_demand': [float(x) for x in balanced_demand],
                'average_load': float(avg_balanced),
                'peak_reduction': float(peak_reduction),
                'load_factor': float(load_factor),
                'distribution_data': distribution_data
            }
            
            return JsonResponse(response_data)
            
        except Exception as e:
            print(f"Error in adjust_demand: {str(e)}")
            return JsonResponse({'error': str(e)}, status=400)
    
    return JsonResponse({'error': 'Invalid request method'}, status=405)

def generate_distribution_data(balanced_demand):
    """
    Generate power distribution data including leakage points and alerts
    """
    # Calculate average demand
    avg_demand = np.mean(balanced_demand)
    
    # Generate time series data for a 24-hour period
    time_series = []
    critical_points = []
    alerts = []
    
    # Define grid segments (simulating different areas of the distribution network)
    grid_segments = [
        {'name': 'Urban Area 1', 'distance': 15, 'base_loss': 0.05},
        {'name': 'Industrial Zone', 'distance': 35, 'base_loss': 0.08},
        {'name': 'Rural Area 1', 'distance': 60, 'base_loss': 0.12},
        {'name': 'Suburban Area', 'distance': 45, 'base_loss': 0.07},
        {'name': 'Rural Area 2', 'distance': 80, 'base_loss': 0.15}
    ]
    
    # Generate input and output power data
    input_power = []
    output_power = []
    power_loss = []
    total_loss = 0
    
    for hour in range(24):
        # Base load varies by hour
        base = avg_demand * (1 + 0.2 * math.sin(hour * math.pi / 12))
        
        # Add some random variation
        input_val = base * (1 + random.uniform(-0.1, 0.1))
        input_power.append(input_val)
        
        # Calculate cumulative loss through grid segments
        current_power = input_val
        segment_losses = []
        
        for segment in grid_segments:
            # Calculate loss for this segment
            base_loss = segment['base_loss']
            distance_factor = segment['distance'] / 50  # Normalize distance
            temperature_factor = 1 + 0.1 * math.sin(hour * math.pi / 12)  # Temperature effect
            
            # Calculate actual loss for this segment
            segment_loss = current_power * base_loss * distance_factor * temperature_factor
            current_power -= segment_loss
            segment_losses.append(segment_loss)
            
            # Check if this is a critical point (high loss)
            if segment_loss > input_val * 0.05:  # More than 5% loss in one segment
                critical_points.append({
                    'distance': segment['distance'],
                    'location': random.uniform(0, 1),  # Random position in the segment
                    'loss': segment_loss,
                    'description': f'High loss detected in {segment["name"]}'
                })
        
        output_power.append(current_power)
        total_loss_hour = input_val - current_power
        power_loss.append(total_loss_hour)
        total_loss += total_loss_hour
    
    # Calculate overall metrics
    total_loss = np.mean(power_loss)
    loss_percentage = (total_loss / np.mean(input_power)) * 100
    
    # Generate alerts based on analysis
    if loss_percentage > 15:
        alerts.append({
            'severity': 'High',
            'message': f'Critical power loss of {loss_percentage:.1f}% detected',
            'recommendation': 'Immediate inspection of transmission lines recommended'
        })
    elif loss_percentage > 10:
        alerts.append({
            'severity': 'Medium',
            'message': f'Elevated power loss of {loss_percentage:.1f}% detected',
            'recommendation': 'Schedule maintenance check within 48 hours'
        })
    
    if len(critical_points) > 3:
        alerts.append({
            'severity': 'High',
            'message': f'Multiple critical points detected ({len(critical_points)} locations)',
            'recommendation': 'Deploy maintenance team for network inspection'
        })
    
    return {
        'input_power': input_power,
        'output_power': output_power,
        'power_loss': power_loss,
        'total_loss': total_loss,
        'loss_percentage': loss_percentage,
        'critical_points': critical_points,
        'alerts': alerts
    }

def calculate_power_leakage(input_power, transmission_distance=100):
    """
    Calculate power leakage based on transmission distance and input power
    Returns leakage in MW and percentage
    """
    # Base transmission loss (usually 6-8% in real grids)
    base_loss_rate = 0.07
    
    # Distance factor (increased loss with distance)
    distance_factor = transmission_distance / 100
    
    # Calculate actual loss rate (increases with distance)
    actual_loss_rate = base_loss_rate * (1 + (distance_factor - 1) * 0.2)
    
    # Calculate losses
    leakage_mw = input_power * actual_loss_rate
    leakage_percentage = actual_loss_rate * 100
    
    return round(leakage_mw, 2), round(leakage_percentage, 2)

def analyze_grid_health(balanced_demand, transmission_distance=100):
    """
    Analyze grid health and identify potential issues
    """
    avg_demand = np.mean(balanced_demand)
    issues = []
    
    # Calculate power leakage
    leakage_mw, leakage_percentage = calculate_power_leakage(avg_demand, transmission_distance)
    
    # Define thresholds
    voltage_fluctuation = random.uniform(-5, 5)  # Simulated voltage fluctuation
    equipment_health = random.uniform(70, 100)   # Simulated equipment health score
    
    # Check for various issues
    if leakage_percentage > 8:
        issues.append({
            'type': 'High Power Leakage',
            'severity': 'High',
            'details': f'Power loss: {leakage_mw:.2f} MW ({leakage_percentage:.1f}%)'
        })
    
    if abs(voltage_fluctuation) > 3:
        issues.append({
            'type': 'Voltage Fluctuation',
            'severity': 'Medium',
            'details': f'Variation: {voltage_fluctuation:.1f}%'
        })
    
    if equipment_health < 85:
        issues.append({
            'type': 'Equipment Stress',
            'severity': 'Medium',
            'details': f'Health Score: {equipment_health:.1f}%'
        })
    
    return {
        'leakage': {
            'amount_mw': leakage_mw,
            'percentage': leakage_percentage,
            'distance_km': transmission_distance
        },
        'voltage_fluctuation': voltage_fluctuation,
        'equipment_health': equipment_health,
        'issues': issues
    }

@login_required(login_url='grid_management:login')
def get_real_time_data(request):
    try:
        # Generate random demand values
        current_demand = random.uniform(0.8, 1.7)
        peak_demand = current_demand * 1.2
        avg_demand = current_demand * 0.9
        
        # Generate simple distribution data
        input_power = [random.uniform(0.6, 1.0) for _ in range(24)]  # Values between 0.6-1.0 MW
        output_power = [x * 0.95 for x in input_power]  # 5% loss
        power_loss = [x - y for x, y in zip(input_power, output_power)]
        
        total_loss = sum(power_loss)
        loss_percentage = (total_loss / sum(input_power)) * 100
        
        # Generate simple leakage points (2-3 points)
        leakage_points = []
        for _ in range(random.randint(2, 3)):
            leakage_points.append({
                'x': random.uniform(0.1, 0.9),  # x coordinate (0-1)
                'y': random.uniform(0.1, 0.9),  # y coordinate (0-1)
                'value': random.uniform(0.02, 0.08)  # leakage value (2-8%)
            })
        
        distribution_data = {
            'input_power': input_power,
            'output_power': output_power,
            'power_loss': power_loss,
            'total_loss': total_loss,
            'loss_percentage': loss_percentage,
            'leakage_points': leakage_points
        }
        
        return JsonResponse({
            'current_demand': round(current_demand, 2),
            'peak_demand': round(peak_demand, 2),
            'avg_demand': round(avg_demand, 2),
            'distribution_data': distribution_data
        })
        
    except Exception as e:
        print(f"Error in get_real_time_data: {str(e)}")
        return JsonResponse({'error': str(e)}, status=500)

@login_required(login_url='grid_management:login')
def get_user_details(request):
    if request.method == 'GET':
        try:
            # Get the user's profile
            user_profile = UserProfile.objects.get(user=request.user)
            
            # Get recent consumption data
            recent_consumption = PowerConsumption.objects.filter(
                user_profile=user_profile
            ).order_by('-timestamp')[:30]  # Last 30 readings
            
            # Calculate average consumption
            avg_consumption = 0
            if recent_consumption:
                avg_consumption = sum(c.actual_consumption for c in recent_consumption) / len(recent_consumption)
            
            # Get recent bills
            recent_bills = Bill.objects.filter(
                user_profile=user_profile
            ).order_by('-generated_date')[:3]  # Last 3 bills
            
            # Format bills data
            bills_data = [{
                'month': bill.month,
                'year': bill.year,
                'units': bill.units_consumed,
                'amount': bill.total_amount,
                'generated_date': bill.generated_date.strftime('%Y-%m-%d')
            } for bill in recent_bills]
            
            return JsonResponse({
                'success': True,
                'data': {
                    'customer_id': user_profile.customer_id,
                    'full_name': user_profile.full_name,
                    'username': request.user.username,
                    'email': request.user.email,
                    'door_number': user_profile.door_number,
                    'area_name': user_profile.area_name,
                    'location': user_profile.location,
                    'service_no': user_profile.service_no,
                    'join_date': request.user.date_joined.strftime('%Y-%m-%d'),
                    'avg_consumption': round(avg_consumption, 2),
                    'recent_bills': bills_data,
                    'last_updated': datetime.now(pytz.timezone('Asia/Kolkata')).strftime('%Y-%m-%d %H:%M:%S')
                }
            })
        except UserProfile.DoesNotExist:
            return JsonResponse({'success': False, 'error': 'User profile not found'})
        except Exception as e:
            return JsonResponse({'success': False, 'error': str(e)})
    return JsonResponse({'success': False, 'error': 'Invalid request method'})

@login_required(login_url='grid_management:login')
def profile_view(request):
    """
    Render the profile page with user details
    """
    try:
        # Get user profile data
        user_profile = UserProfile.objects.get(user=request.user)
        
        # Get power consumption statistics
        consumption_data = PowerConsumption.objects.filter(
            user_profile=user_profile
        ).order_by('-timestamp')
        
        # Calculate consumption trends
        if consumption_data:
            monthly_consumption = {}
            for record in consumption_data:
                month_key = record.timestamp.strftime('%Y-%m')
                if month_key not in monthly_consumption:
                    monthly_consumption[month_key] = []
                monthly_consumption[month_key].append(record.actual_consumption)
            
            # Calculate monthly averages
            monthly_averages = {
                month: sum(values) / len(values)
                for month, values in monthly_consumption.items()
            }
        else:
            monthly_averages = {}
        
        context = {
            'user_profile': user_profile,
            'monthly_averages': monthly_averages,
            'recent_consumption': consumption_data[:30] if consumption_data else [],
            'page_title': 'User Profile',
            'active_tab': 'profile'
        }
        
        return render(request, 'grid_management/profile.html', context)
        
    except Exception as e:
        print(f"Error in profile view: {str(e)}")
        context = {
            'error_message': 'Failed to load profile data',
            'page_title': 'Error',
            'active_tab': 'profile'
        }
        return render(request, 'grid_management/profile.html', context)

@login_required(login_url='grid_management:login')
def get_user_details_by_id(request):
    customer_id = request.GET.get('customer_id', '')
    if not customer_id:
        return JsonResponse({
            'success': False,
            'error': 'Customer ID is required'
        })
        
    try:
        user_profile = UserProfile.objects.get(customer_id=customer_id)
        # Calculate previous bills count and total amount
        previous_bills = Bill.objects.filter(user_profile=user_profile).order_by('-generated_date')
        last_bill = previous_bills.first()
        previous_bills_data = [{
           'total_amount': bill.total_amount,
           'month': bill.month,
           'year': bill.year,
           'units_consumed': bill.units_consumed,
           'rate_per_unit': bill.rate_per_unit,
           'amount': bill.amount,
           'tax': bill.tax,
           'generated_date': bill.generated_date
        } for bill in previous_bills]

        
        return JsonResponse({
                            'success': True,
                                 'data': {
                                 'full_name': user_profile.full_name,
                           'service_no': user_profile.service_no,
                         'door_number': user_profile.door_number,
        'area_name': user_profile.area_name,
        'location': user_profile.location,
        'last_bill_amount': last_bill.total_amount if last_bill else 0,
        'last_bill_units': last_bill.units_consumed if last_bill else 0,
        'last_bill_month': last_bill.month if last_bill else '',
        'last_bill_year': last_bill.year if last_bill else '',
                               'total_bills': previous_bills.count(),
                             'previous_bills_data': previous_bills_data
                             }
                        })
    except UserProfile.DoesNotExist:
        return JsonResponse({
            'success': False,
            'error': f'No customer found with ID: {customer_id}'
        })
    except Exception as e:
        return JsonResponse({
            'success': False,
            'error': str(e)
        })
    
@login_required(login_url='grid_management:login')    
def get_previous_bills_data(request):
    """
    Fetches and formats the previous bills data for the given user profile.
    """
    customer_id = request.GET.get('customer_id', '')
    
    if not customer_id:
        return JsonResponse({
            'success': False,
            'error': 'Customer ID is required'
        })

    try:
        user_profile = UserProfile.objects.get(customer_id=customer_id)
    except UserProfile.DoesNotExist:
        return JsonResponse({
            'success': False,
            'error': 'Customer not found'
        })

    # Fetch previous bills for the user
    previous_bills = Bill.objects.filter(user_profile=user_profile).order_by('-generated_date')
    
    # Format the bills data
    previous_bills_data = [{
        'total_amount': bill.total_amount,
        'month': bill.month,
        'year': bill.year,
        'units_consumed': bill.units_consumed,
        'rate_per_unit': bill.rate_per_unit,
        'amount': bill.amount,
        'tax': bill.tax,
        'generated_date': bill.generated_date
    } for bill in previous_bills]

    # Aggregate data for the latest bill and total count
    last_bill = previous_bills.first()
    last_bill_amount = last_bill.total_amount if last_bill else 0
    last_bill_units = last_bill.units_consumed if last_bill else 0
    last_bill_period = f"{last_bill.month} {last_bill.year}" if last_bill else 'N/A'
    total_bills = previous_bills.count()

    return JsonResponse({
        'success': True,
        'data': {
            'last_bill_amount': last_bill_amount,
            'last_bill_units': last_bill_units,
            'last_bill_period': last_bill_period,
            'total_bills': total_bills,
            'previous_bills_data': previous_bills_data
        }
    })


@login_required(login_url='grid_management:login')
def bill_generation(request):
    if request.method == 'POST':
        try:
            # Get user details
            service_no = request.POST.get('service_no')
            user_profile = UserProfile.objects.get(service_no=service_no)
            
            # Get bill details
            units_consumed = float(request.POST.get('units_consumed', 0))
            billing_period = request.POST.get('billing_period', 'monthly')
            
            # Get current month and year
            current_date = datetime.now()
            month = current_date.month
            year = current_date.year
            
            # Calculate bill amount
            rate_per_unit = 8  # Base rate per unit
            if units_consumed > 500:
                rate_per_unit = 10
            elif units_consumed > 200:
                rate_per_unit = 9
            
            # Adjust units based on billing period
            if billing_period == 'quarterly':
                units_consumed = units_consumed / 3
            elif billing_period == 'yearly':
                units_consumed = units_consumed / 12
                
            amount = units_consumed * rate_per_unit
            tax = amount * 0.05  # 5% tax
            total_amount = amount + tax
            
            # Create bill object
            bill = Bill.objects.create(
                user_profile=user_profile,
                month=month,
                year=year,
                units_consumed=units_consumed,
                rate_per_unit=rate_per_unit,
                amount=amount,
                tax=tax,
                total_amount=total_amount
            )
            
            return JsonResponse({
                'success': True,
                'bill_id': bill.id,
                'amount': amount,
                'tax': tax,
                'total_amount': total_amount,
                'customer_name': user_profile.user.get_full_name(),
                'customer_id': user_profile.customer_id,
                'service_no': service_no,
                'billing_period': billing_period,
                'units_consumed': units_consumed,
                'rate_per_unit': rate_per_unit
            })
            
        except UserProfile.DoesNotExist:
            return JsonResponse({'success': False, 'error': 'User not found'})
        except ValueError as e:
            return JsonResponse({'success': False, 'error': 'Invalid input values'})
        except Exception as e:
            return JsonResponse({'success': False, 'error': str(e)})
            
    return render(request, 'grid_management/bill_generation.html')

@login_required(login_url='grid_management:login')
def get_consumption_stats(request):
    try:
        # Get the current user's profile
        user_profile = UserProfile.objects.get(user=request.user)
        
        # Get current time
        current_time = datetime.now()
        
        # Get recent consumption data for calculations
        recent_consumption = PowerConsumption.objects.filter(
            user_profile=user_profile,
            timestamp__date=current_time.date()
        ).order_by('-timestamp')

        # Calculate daily average
        daily_target = 30.0  # kWh, can be customized per user
        if recent_consumption.exists():
            daily_average = recent_consumption.aggregate(Avg('actual_consumption'))['actual_consumption__avg']
        else:
            daily_average = 0.0

        # Calculate monthly usage
        monthly_consumption = PowerConsumption.objects.filter(
            user_profile=user_profile,
            timestamp__year=current_time.year,
            timestamp__month=current_time.month
        )
        monthly_target = daily_target * current_time.day
        if monthly_consumption.exists():
            monthly_usage = monthly_consumption.aggregate(Sum('actual_consumption'))['actual_consumption__sum']
        else:
            monthly_usage = 0.0

        # Calculate peak hours usage (6 PM - 10 PM)
        peak_consumption = PowerConsumption.objects.filter(
            user_profile=user_profile,
            timestamp__date=current_time.date(),
            timestamp__hour__range=(18, 21)  # 6 PM to 10 PM
        )
        if peak_consumption.exists():
            peak_usage = peak_consumption.aggregate(Sum('actual_consumption'))['actual_consumption__sum']
        else:
            peak_usage = 0.0

        # Calculate carbon footprint (kg CO2 per kWh)
        carbon_factor = 0.5  # kg CO2 per kWh, varies by region
        carbon_footprint = monthly_usage * carbon_factor if monthly_usage else 0.0
        
        data = {
            'daily_average': daily_average or 0.0,
            'daily_target': daily_target,
            'monthly_usage': monthly_usage or 0.0,
            'monthly_target': monthly_target,
            'peak_usage': peak_usage or 0.0,
            'carbon_footprint': carbon_footprint or 0.0
        }
        
        return JsonResponse(data)
        
    except UserProfile.DoesNotExist:
        return JsonResponse({'error': 'User profile not found'}, status=404)
    except Exception as e:
        return JsonResponse({'error': str(e)}, status=500)
