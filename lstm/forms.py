from django import forms

FRUIT_CHOICES = [
    ('orange', 'Oranges'),
    ('cantaloupe', 'Cantaloupes'),
    ('mango', 'Mangoes'),
    ('honeydew', 'Honeydews'),
    ]

NOTE_CHOICES = [
    ("do", "Do"),
    ("re", "Re"),
    ("mi", "Mi"),
    ("fa", "Fa"),
    ("sol", "Sol"),
    ("la", "La"),
    ("si", "Si")
]

class LSTMForm(forms.Form):
    trained_lstm_selected = forms.CharField(label='Choose a trained model', widget=forms.Select(choices=FRUIT_CHOICES))

class InputNoteForm(forms.Form):
    input_note = forms.CharField(label='Choose a note', widget=forms.Select(choices=NOTE_CHOICES))
