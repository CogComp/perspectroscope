# Generated by Django 2.1.7 on 2019-08-12 16:33

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('app', '0004_auto_20190612_0241'),
    ]

    operations = [
        migrations.AddField(
            model_name='feedbackrecord',
            name='comment',
            field=models.TextField(default=''),
        ),
        migrations.AddField(
            model_name='feedbackrecord',
            name='stance',
            field=models.CharField(default='UNK', max_length=3),
        ),
        migrations.AddField(
            model_name='feedbackrecord',
            name='username',
            field=models.CharField(default='Anonymous', max_length=100),
        ),
    ]
