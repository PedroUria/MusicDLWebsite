from django import forms

FRUIT_CHOICES = [
    ('orange', 'Oranges'),
    ('cantaloupe', 'Cantaloupes'),
    ('mango', 'Mangoes'),
    ('honeydew', 'Honeydews'),
    ]

class LSTMForm(forms.Form):
    trained_lstm_selected = forms.CharField(label='Choose a trained model', widget=forms.Select(choices=FRUIT_CHOICES))

class RightNoteThresForm(forms.Form):
    thres_note_selected_right = forms.FloatField(label="Right Hand Notes Probability Threshold", min_value=0, max_value=1, initial=0.9)

class LeftNoteThresForm(forms.Form):
    thres_note_selected_left = forms.FloatField(label="Left Hand Notes Probability Threshold", min_value=0, max_value=1, initial=0.9)

class RightHoldThresForm(forms.Form):
    thres_hold_selected_right = forms.FloatField(label="Right Hand Hold Probability Threshold", min_value=0, max_value=1, initial=0.7)

class LeftHoldThresForm(forms.Form):
    thres_hold_selected_left = forms.FloatField(label="Left Hand Hold Probability Threshold", min_value=0, max_value=1, initial=0.7)

class SeqLenForm(forms.Form):
    seq_len_selected = forms.IntegerField(label="Input Sequence Length", min_value=1, max_value=3000, initial=50)

class NStepsForm(forms.Form):
    n_steps_selected = forms.IntegerField(label="Notes to Generate", min_value=1, max_value=3000, initial=300)

class TempoForm(forms.Form):
    tempo_selected = forms.IntegerField(label="Tempo", min_value=1, max_value=300, initial=74)
    # TODO: Figure out min and max values for tempo
