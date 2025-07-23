from django.http import JsonResponse
from datetime import datetime, timedelta
import pytz
from django.contrib.auth.decorators import login_required
import firebase_admin
from firebase_admin import credentials, firestore

# Initialize Firebase
cred = credentials.Certificate("smart_grid/grid_management_backup_$(date +%Y%m%d_%H%M%S)/fireservice.json")  # Path to your Firebase credentials
firebase_admin.initialize_app(cred)
db = firestore.client()

def calculate_bill_amount(energy_consumed, billing_period):
    rate_slabs = {
        'monthly': [
            {'limit': 100, 'rate': 3.50},
            {'limit': 300, 'rate': 4.50},
            {'limit': 500, 'rate': 6.00},
            {'limit': float('inf'), 'rate': 7.50}
        ],
        'quarterly': [
            {'limit': 300, 'rate': 3.25},
            {'limit': 900, 'rate': 4.25},
            {'limit': 1500, 'rate': 5.75},
            {'limit': float('inf'), 'rate': 7.25}
        ],
        'yearly': [
            {'limit': 1200, 'rate': 3.00},
            {'limit': 3600, 'rate': 4.00},
            {'limit': 6000, 'rate': 5.50},
            {'limit': float('inf'), 'rate': 7.00}
        ]
    }

    slabs = rate_slabs[billing_period]
    
    remaining_units = energy_consumed
    total_amount = 0
    effective_rate = 0
    
    for slab in slabs:
        if remaining_units <= 0:
            break
            
        units_in_slab = min(remaining_units, slab['limit'])
        amount_in_slab = units_in_slab * slab['rate']
        
        total_amount += amount_in_slab
        remaining_units -= units_in_slab
    
    effective_rate = round(total_amount / energy_consumed, 2)
    
    return total_amount, effective_rate

@login_required
def generate_bill(request):
    if request.method == 'POST':
        try:
            from .models import UserProfile, Bill
            
            # Get user details
            service_no = request.POST.get('service_no')
            user_profile = UserProfile.objects.get(service_no=service_no)
            
            # Get bill details
            units_consumed = float(request.POST.get('units_consumed', 0))
            billing_period = request.POST.get('billing_period', 'monthly')
            
            # Get current date
            current_date = datetime.now()
            
            # Calculate bill amount using the rate slabs
            amount, effective_rate = calculate_bill_amount(units_consumed, billing_period)
            
            # Calculate tax (5%)
            tax = amount * 0.05
            total_amount = amount + tax
            
            # Create bill object
            bill = Bill.objects.create(
                user_profile=user_profile,
                month=current_date.month,
                year=current_date.year,
                units_consumed=units_consumed,
                rate_per_unit=effective_rate,
                amount=amount,
                tax=tax,
                total_amount=total_amount
            )
            
            # Calculate due date (15 days from bill generation)
            due_date = current_date + timedelta(days=15)
            
            # Prepare bill data for Firebase
            bill_data = {
                'customer_id': user_profile.customer_id,
                'customer_name': user_profile.user.get_full_name(),
                'service_no': service_no,
                'billing_period': billing_period,
                'units_consumed': units_consumed,
                'rate_per_unit': effective_rate,
                'amount': amount,
                'tax': tax,
                'total_amount': total_amount,
                'generation_date': current_date.strftime('%d-%m-%Y'),
                'due_date': due_date.strftime('%d-%m-%Y')
            }
            
            # Store bill in Firebase Firestore
            bill_ref = db.collection('bills').add(bill_data)
            
            return JsonResponse({
                'success': True,
                'bill_id': bill.id,
                'customer_name': user_profile.user.get_full_name(),
                'customer_id': user_profile.customer_id,
                'service_no': service_no,
                'billing_period': billing_period,
                'units_consumed': units_consumed,
                'rate_per_unit': effective_rate,
                'amount': amount,
                'tax': tax,
                'total_amount': total_amount,
                'generation_date': current_date.strftime('%d-%m-%Y'),
                'due_date': due_date.strftime('%d-%m-%Y')
            })
            
        except UserProfile.DoesNotExist:
            return JsonResponse({'success': False, 'error': 'User not found'})
        except ValueError as e:
            return JsonResponse({'success': False, 'error': 'Invalid input values'})
        except Exception as e:
            return JsonResponse({'success': False, 'error': str(e)})
            
    return JsonResponse({'success': False, 'error': 'Invalid request method'})
