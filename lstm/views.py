from django.shortcuts import render, get_object_or_404

from .models import Author
from .forms import LSTMForm
from django import forms

from django.template import RequestContext

import os


def home(request):
    authors = Author.objects.all()
    return render(request, 'home.html', {"authors": authors})

def author(request, pk):
    try:
        lstm_selected = request.GET["trained_lstm_selected"]
        samples = [sample for sample in os.listdir(os.getcwd() + "/static/lstm/audio") if
                   lstm_selected.replace("pkl", "mp3") in sample]
    except:
        samples = "No selection"
    author = get_object_or_404(Author, pk=pk)
    lstm_trained_choices = []
    for lstm_trained in author.lstms_trained:
        lstm_trained_choices.append((lstm_trained.lower(), lstm_trained))
    form = LSTMForm()
    form.fields["trained_lstm_selected"] = forms.CharField(label='Choose a trained model', widget=forms.Select(choices=lstm_trained_choices))
    return render(request, 'author.html', {"author": author, "form": form, "samples": samples})


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
