# Generated by Django 5.1.3 on 2024-12-07 14:12

import django.db.models.deletion
from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('grid_management', '0004_userprofile_customer_id'),
    ]

    operations = [
        migrations.CreateModel(
            name='LoadBalancingMetrics',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('timestamp', models.DateTimeField(auto_now_add=True)),
                ('average_load', models.FloatField()),
                ('peak_load', models.FloatField()),
                ('load_factor', models.FloatField()),
                ('peak_reduction', models.FloatField()),
                ('total_demand', models.FloatField()),
                ('balanced_demand', models.FloatField()),
            ],
            options={
                'ordering': ['-timestamp'],
                'indexes': [models.Index(fields=['timestamp'], name='grid_manage_timesta_1d8316_idx')],
            },
        ),
        migrations.CreateModel(
            name='PowerConsumption',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('timestamp', models.DateTimeField()),
                ('actual_consumption', models.FloatField()),
                ('predicted_consumption', models.FloatField()),
                ('temperature', models.FloatField()),
                ('solar_radiation', models.FloatField()),
                ('wind_speed', models.FloatField()),
                ('humidity', models.FloatField()),
                ('cloud_cover', models.FloatField()),
                ('theft_detected', models.BooleanField(default=False)),
                ('theft_probability', models.FloatField(default=0.0)),
                ('user_profile', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to='grid_management.userprofile')),
            ],
            options={
                'ordering': ['-timestamp'],
                'indexes': [models.Index(fields=['user_profile', 'timestamp'], name='grid_manage_user_pr_039470_idx'), models.Index(fields=['theft_detected'], name='grid_manage_theft_d_02f188_idx')],
            },
        ),
    ]
