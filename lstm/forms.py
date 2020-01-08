from django import forms

FRUIT_CHOICES = [
    ('orange', 'Oranges'),
    ('cantaloupe', 'Cantaloupes'),
    ('mango', 'Mangoes'),
    ('honeydew', 'Honeydews'),
    ]

class LSTMForm(forms.Form):
    trained_lstm_selected = forms.CharField(label='Choose a trained model', widget=forms.Select(choices=FRUIT_CHOICES))
