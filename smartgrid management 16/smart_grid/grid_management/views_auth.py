from django.shortcuts import render, redirect
from django.contrib.auth import login, authenticate, logout
from django.contrib.auth.forms import UserCreationForm
from django.contrib import messages
from django.contrib.auth.decorators import login_required
from django.contrib.auth.models import User
from .models import UserProfile

def signup_view(request):
    if request.method == 'POST':
        username = request.POST.get('username')
        full_name = request.POST.get('full_name')
        email = request.POST.get('email')
        password1 = request.POST.get('password1')
        password2 = request.POST.get('password2')
        door_number = request.POST.get('door_number')
        area_name = request.POST.get('area_name')
        location = request.POST.get('location')
        service_no = request.POST.get('service_no')

        if not full_name:
            messages.error(request, 'Full name is required!')
            return redirect('grid_management:signup')

        if password1 != password2:
            messages.error(request, 'Passwords do not match!')
            return redirect('grid_management:signup')

        if User.objects.filter(username=username).exists():
            messages.error(request, 'Username already exists!')
            return redirect('grid_management:signup')

        if User.objects.filter(email=email).exists():
            messages.error(request, 'Email already registered!')
            return redirect('grid_management:signup')

        if UserProfile.objects.filter(service_no=service_no).exists():
            messages.error(request, 'Service number already registered!')
            return redirect('grid_management:signup')

        try:
            # Create the user
            user = User.objects.create_user(
                username=username,
                email=email,
                password=password1,
                first_name=full_name.split()[0],
                last_name=' '.join(full_name.split()[1:]) if len(full_name.split()) > 1 else ''
            )
            
            # Create the user profile
            UserProfile.objects.create(
                user=user,
                full_name=full_name,
                door_number=door_number,
                area_name=area_name,
                location=location,
                service_no=service_no
            )
            
            messages.success(request, 'Account created successfully! Please login.')
            return redirect('grid_management:login')
        except Exception as e:
            if user:
                user.delete()  # Rollback user creation if profile creation fails
            messages.error(request, f'Error creating account: {str(e)}')
            return redirect('grid_management:signup')

    return render(request, 'signup.html')

def login_view(request):
    if request.method == 'POST':
        username = request.POST.get('username')
        password = request.POST.get('password')
        user = authenticate(request, username=username, password=password)
        
        if user is not None:
            login(request, user)
            messages.success(request, f'Welcome back, {username}!')
            return redirect('grid_management:index')
        else:
            messages.error(request, 'Invalid username or password!')
            return redirect('grid_management:login')

    return render(request, 'login.html')

def logout_view(request):
    logout(request)
    messages.info(request, 'You have been logged out successfully!')
    return redirect('grid_management:login')
