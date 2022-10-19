from django import forms
from django.contrib.auth.forms import UserCreationForm
from django.contrib.auth.models import User


class SignUpForm(UserCreationForm):
    first_name = forms.CharField(max_length=30, required=True)
    last_name = forms.CharField(max_length=30, required=True)
    email = forms.EmailField(max_length=254)
    password1 = forms.CharField(label='Enter password',
                                    widget=forms.PasswordInput)
    password2 = forms.CharField(label='Confirm password',
                                    widget=forms.PasswordInput)
    class Meta:
        model = User
        help_texts = {'password2': None, 'email': None, 'username':None }
        fields = ('username', 'first_name', 'last_name', 'email', 'password1', 'password2', )


class LoginForm(forms.Form):
    username = forms.CharField(max_length=63)
    password = forms.CharField(max_length=63, widget=forms.PasswordInput)
