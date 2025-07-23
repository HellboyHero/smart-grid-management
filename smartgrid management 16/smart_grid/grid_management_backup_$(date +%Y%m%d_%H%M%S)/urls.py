from django.urls import path
from . import views
from . import views_billing
from . import views_auth

app_name = 'grid_management'

urlpatterns = [
    path('', views.index, name='index'),
    path('predict/', views.predict, name='predict'),
    path('adjust_demand/', views.adjust_demand, name='adjust_demand'),
    path('get_real_time_data/', views.get_real_time_data, name='get_real_time_data'),
    path('generate_bill/', views_billing.generate_bill, name='generate_bill'),
    path('bill_generation/', views.bill_generation, name='bill_generation'),
    path('get_user_details/', views.get_user_details, name='get_user_details'),
    path('get_user_details_by_id/', views.get_user_details_by_id, name='get_user_details_by_id'),
    path('login/', views_auth.login_view, name='login'),
    path('signup/', views_auth.signup_view, name='signup'),
    path('logout/', views_auth.logout_view, name='logout'),
    path('profile/', views.profile_view, name='profile'),
    path('get_consumption_stats/', views.get_consumption_stats, name='get_consumption_stats'),
]
