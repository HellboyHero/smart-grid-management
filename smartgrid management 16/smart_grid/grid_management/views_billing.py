from django.http import JsonResponse
from datetime import datetime, timedelta
import pytz
from django.contrib.auth.decorators import login_required

def calculate_bill_amount(energy_consumed, billing_period):
    # Define rate slabs (in kWh) and their prices
    rate_slabs = {
        'monthly': [
            {'limit': 100, 'rate': 3.50},   # First 100 units
            {'limit': 300, 'rate': 4.50},   # 101-300 units
            {'limit': 500, 'rate': 6.00},   # 301-500 units
            {'limit': float('inf'), 'rate': 7.50}  # Above 500 units
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

    # Get rate slabs for the billing period
    slabs = rate_slabs[billing_period]
    
    # Calculate total amount
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
    
    # Calculate effective rate per kWh
    effective_rate = round(total_amount / energy_consumed, 2)
    
    return total_amount, effective_rate

@login_required
def generate_bill(request):
    if request.method == 'POST':
        try:
            customer_id = request.POST.get('customer_id')
            customer_name = request.POST.get('customer_name')
            service_no = request.POST.get('service_no')
            billing_period = request.POST.get('billing_period')
            energy_consumed = float(request.POST.get('energy_consumed'))
            
            # Validate service number
            if not service_no or len(service_no) != 13 or not service_no.isdigit():
                return JsonResponse({'error': 'Invalid service number. Must be exactly 13 digits.'})
            
            # Calculate bill amount
            base_amount, rate_per_kwh = calculate_bill_amount(energy_consumed, billing_period)
            
            # Calculate tax (assuming 5% GST)
            tax_rate = 5
            tax_amount = round(base_amount * tax_rate / 100, 2)
            total_amount = round(base_amount + tax_amount, 2)
            
            # Generate dates
            ist = pytz.timezone('Asia/Kolkata')
            generation_date = datetime.now(ist).strftime('%d-%m-%Y')
            due_date = (datetime.now(ist) + timedelta(days=15)).strftime('%d-%m-%Y')
            
            response_data = {
                'customer_id': customer_id,
                'customer_name': customer_name,
                'service_no': service_no,
                'billing_period': billing_period.capitalize(),
                'energy_consumed': energy_consumed,
                'rate_per_kwh': rate_per_kwh,
                'base_amount': round(base_amount, 2),
                'tax_rate': tax_rate,
                'tax_amount': tax_amount,
                'total_amount': total_amount,
                'generation_date': generation_date,
                'due_date': due_date
            }
            
            return JsonResponse(response_data)
        except Exception as e:
            return JsonResponse({'error': str(e)}, status=500)
            
    return JsonResponse({'error': 'Invalid request method'}, status=405)

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
