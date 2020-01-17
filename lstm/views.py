from django.shortcuts import render, get_object_or_404

from .models import Author
from .forms import LSTMForm, RightNoteThresForm, LeftNoteThresForm, RightHoldThresForm, LeftHoldThresForm, SeqLenForm, NStepsForm, TempoForm
from django import forms

import os
from .utils_generate import generate_music
from midi2audio import FluidSynth


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
        input_notes_seq = request.GET["submitted"]
        thres_note_selected_right, thres_hold_selected_right = float(request.GET["thres_note_selected_right"]), float(request.GET["thres_hold_selected_right"])
        thres_note_selected_left, thres_hold_selected_left = float(request.GET["thres_note_selected_left"]), float(request.GET["thres_hold_selected_left"])
        seq_len_selected, n_steps_selected = int(request.GET["seq_len_selected"]), int(request.GET["n_steps_selected"])
        tempo_selected = int(request.GET["tempo_selected"])
        current_midi_and_wav_files = [file for file in os.listdir(os.getcwd() + "/static/lstm/generated_samples/") if ".mid" in file or ".wav" in file]
        for file in current_midi_and_wav_files:
            os.remove(os.getcwd() + "/static/lstm/generated_samples/" + file)
        output_file_name = os.getcwd() + "/static/lstm/generated_samples/" + trained_lstm_selected.replace(".pkl", "") + "_generated"
        generate_music(os.getcwd() + "/static/lstm/neuralnetworks/" + trained_lstm_selected, input_notes_seq,
                       output_file_name, hold_thres=[thres_hold_selected_left, thres_hold_selected_right],
                       note_thres=[thres_note_selected_left, thres_note_selected_right], seq_len=seq_len_selected,
                       n_steps=n_steps_selected, tempo=tempo_selected, both_hands=True if "melody" not in trained_lstm_selected else False)
        FluidSynth(os.getcwd() + "/static/lstm/generated_samples/PianoSoundfonts/" + "FullGrandPiano.sf2").midi_to_audio(
            output_file_name + ".mid", output_file_name + ".wav")
    except Exception as e:
        input_notes_seq = ""  # str(e)
        output_file_name = ""
    try:
        clear = request.GET["clear"]
    except:
        clear = ""
    author = get_object_or_404(Author, pk=pk)
    form_note_thres_right, form_hold_thres_right = RightNoteThresForm(), RightHoldThresForm()
    form_note_thres_left, form_hold_thres_left = LeftNoteThresForm(), LeftHoldThresForm()
    form_seq_len, form_n_steps, form_tempo = SeqLenForm(), NStepsForm(), TempoForm()
    return render(request, 'author_generate.html',
                  {"author": author, "trained_lstm_selected": trained_lstm_selected,
                   "output_wav": output_file_name[output_file_name.find("static")-1:] + ".wav",
                   "input_notes_seq": input_notes_seq, "clear": clear,
                   "output_mid_name": trained_lstm_selected.replace(".pkl", "") + "_generated.mid",
                   "form_note_thres_right": form_note_thres_right, "form_hold_thres_right": form_hold_thres_right,
                   "form_note_thres_left": form_note_thres_left, "form_hold_thres_left": form_hold_thres_left,
                   "form_seq_len": form_seq_len, "form_n_steps": form_n_steps, "form_tempo": form_tempo},)

def about(request):
    return render(request, 'about.html')
