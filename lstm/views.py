from django.shortcuts import render, get_object_or_404

from .models import Author
from .forms import LSTMForm, InputNoteForm
from django import forms

import os
from .utils_generate import generate_music


def home(request):
    authors = Author.objects.all()
    return render(request, 'home.html', {"authors": authors})

def author(request, pk):
    try:
        lstm_selected = request.GET["trained_lstm_selected"]
        samples = [sample for sample in os.listdir(os.getcwd() + "/static/lstm/audio") if
                   lstm_selected.replace("pkl", "mp3") in sample]
    except:
        lstm_selected = ""
        samples = "No selection"
    author = get_object_or_404(Author, pk=pk)
    lstm_trained_choices = []
    for lstm_trained in author.lstms_trained:
        lstm_trained_choices.append((lstm_trained.lower(), lstm_trained))
    form = LSTMForm()
    form.fields["trained_lstm_selected"] = forms.CharField(label='Choose a trained model', widget=forms.Select(choices=lstm_trained_choices))
    return render(request, 'author.html', {"author": author, "form": form, "samples": samples, "lstm_selected": lstm_selected})


def author_generate(request, pk, trained_lstm_selected):
    try:
        note_selected = request.GET["input_note"]
        generate_music(os.getcwd() + "/static/lstm/neuralnetworks/" + trained_lstm_selected, note_selected,
                        os.getcwd() + "/static/lstm/generated_samples/" + trained_lstm_selected.replace(".pkl", "") + "_generated")
    except Exception as e:
        note_selected = ""  # str(e)
    author = get_object_or_404(Author, pk=pk)
    form = InputNoteForm()
    return render(request, 'author_generate.html', {"author": author, "trained_lstm_selected": trained_lstm_selected, "note_selected": note_selected, "form": form})

"""from .models import LSTM, Samples

def home(request):
    networks = LSTM.objects.all()
    networks_dict = {}
    for network in networks:
        if network.author not in networks_dict:
            networks_dict[network.author] = [network.name]
        else:
            networks_dict[network.author].append(network.name)
    return render(request, 'home.html', {"networks": networks, 'networks_dict': networks_dict})

def lstm_trained(request, pk):
    lstm_trained_network = LSTM.objects.get(pk=pk)
    samples = Samples.objects.all()
    return render(request, 'lstm_trained_network.html', {'lstm_trained_network': lstm_trained_network, "samples": samples})"""
