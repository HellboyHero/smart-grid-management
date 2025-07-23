def profile_view(request):
    user_profile = get_object_or_404(UserProfile, user=request.user)
    bills = Bill.objects.filter(user=request.user).order_by('-billing_date')
    
    context = {
        'user_profile': user_profile,
        'bills': bills,
        'page_title': 'Profile'
    }
    return render(request, 'grid_management/profile.html', context) 