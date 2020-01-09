import music21 as ms  # python3 -m pip install --user music21 for installing on ubuntu instance
import numpy as np
import torch
import torch.nn as nn
from .utils_encoder_decoder import decode
from random import randint
import os
from midi2audio import FluidSynth


class LSTMMusic(nn.Module):

    """
    LSTM network that will try to learn the pattern within a series
    of musical pieces. It consists on a single LSTM layer followed
    by a fully connected Output Layer with a Sigmoid activation function
    """

    def __init__(self, input_size, hidden_size):
        super(LSTMMusic, self).__init__()
        # Input of shape (seq_len, batch_size, input_size)
        self.lstm = nn.LSTM(input_size, hidden_size)
        # Fully connected Layer at the end, output_size=input_size because we want to predict
        self.out = nn.Linear(hidden_size, input_size)  # the next note/sequence of notes
        # We use a Sigmoid activation function instead of the usual Softmax
        # because we want to predict potentially more than one label per vector,
        # like for example, when we have a hold or a chord
        # Idea from: https://www.depends-on-the-definition.com/guide-to-multi-label-classification-with-neural-networks/
        self.act = nn.Sigmoid()

    def forward(self, x, h_c_state):
        y_pred, h_c_state = self.lstm(x, h_c_state)
        return self.act(self.out(y_pred)), h_c_state


def combine(left, right, filepath='test.mid'):
    """
    Input: left - the stream of notes played by the
                pianist with his left hand
            right - the stream of notes played by the
                pianist with his right hand
            filepath - path to the file in which o/p
                midi stream is to be stored
    Output: N/A

    The combines the streams from the left stream and right
    stream and outputs a single stream object
    """
    sc = ms.stream.Stream()
    sc.append(right)
    sc.append(left)
    left.offset = 0.0
    sc.write("midi", filepath + ".mid")
    # breakpoint()
    fs = FluidSynth()
    fs.midi_to_audio(filepath + ".mid", filepath + ".wav")


def get_tempo_dim_back(notes, tempo=74):
    """
    Adds an extra dimension for the tempo
    :param notes: encoded matrix without the tempo dim
    :param tempo: value of the tempo to include
    :return: Same matrix with tempo dimension, in order
    to decode it successfully
    """
    c = np.empty((notes.shape[0], notes.shape[1]+1))
    for idx in range(notes.shape[0]):
        c[idx] = np.hstack((notes[idx], np.array([tempo])))
    return c


def ltsm_gen(net, seq_len, file_name, sampling_idx=0, sequence_start=0, n_steps=100, hidden_size=178,
             time_step=0.05, changing_note=False, note_stuck=False, remove_extra_rests=True):

    """
    Uses the trained LSTM to generate new notes and saves the output to a MIDI file
    This approach uses a whole sequence of notes of one of the pieces we used to train
    the network, with length seq_len, which should be the same as the one used when training
    :param net: Trained LSTM
    :param seq_len: Length of input sequence
    :param file_name: Name to be given to the generated MIDI file
    :param sampling_idx: File to get the input sequence from, out of the pieces used to train the LSTM
    :param sequence_start: Index of the starting sequence, default to 0
    :param n_steps: Number of vectors to generate
    :param hidden_size: Hidden size of the trained LSTM
    :param time_step: Vector duration. Should be the same as the one on get_right_hand()
    :param changing_note: To sample from different sources at some point of the generation
    and add this new note to the sequence. This is done in case the generation gets stuck
    repeating a particular sequence over and over.
    :param note_stuck: To change the note if the generation gets stuck playing the same
    note over and over.
    :param remove_extra_rests: If the generation outputs a lot of rests in between, use this
    :return: None. Just saves the generated music as a .mid file
    """

    notes = []  # Will contain a sequence of the predicted notes
    # x = notes_encoded[sampling_idx][sequence_start:sequence_start+seq_len]  # Uses the input sequence  # TODO
    for nt in x:  # To start predicting. This will be later removed from
        notes.append(nt.cpu().numpy())  # the final output
    h_state = torch.zeros(1, 1, hidden_size).float()
    c_state = torch.zeros(1, 1, hidden_size).float()
    print_first = True  # To print out a message if every component of a
    # predicted vector is less than 0.9
    change_note = False

    for _ in range(n_steps):
        chosen = False  # To account for when no dimension's probability is bigger than 0.9
        y_pred, h_c_state = net(x, (h_state, c_state))  # Predicts the next notes for all
        h_state, c_state = h_c_state[0].data, h_c_state[1].data # the notes in the input sequence
        y_pred = y_pred.data  # We only care about the last predicted note
        y_pred = y_pred[-1]  # (next note after last note of input sequence)
        choose = torch.zeros((1, 1, 178))  # Coverts the probabilities to the actual note vector
        y_pred_left = y_pred[:, :89]
        for idx in range(89):
            if y_pred_left[:, idx] > 0.9:
                choose[:, :, idx] = 1
                chosen = True
        if y_pred_left[:, -1] >= 0.7:  # We add a hold condition, in case the probability
            choose[:, :, 88] = 1  # of having a hold is close to the one of having the pitch
        if not chosen:
            if print_first:
                print("\nPrinting out the maximum prob of all notes for a time step",
                      "when this maximum prob is less than 0.9")
                print_first = False
            pred_note_idx = np.argmax(y_pred_left.cpu())
            choose[:, :, pred_note_idx] = 1
            if pred_note_idx != 87:  # No holds for rests
                if y_pred_left[:, pred_note_idx] - y_pred_left[:, -1] <= 0.2:  # Hold condition
                    choose[:, :, 88] = 1
            print(_, "left", y_pred_left[:, np.argmax(y_pred_left.cpu())])  # Maximum probability out of all components
        y_pred_right = y_pred[:, 89:]
        for idx in range(89):
            if y_pred_right[:, idx] > 0.9:
                choose[:, :, idx + 89] = 1
                chosen = True
        if y_pred_right[:, -1] >= 0.7:
            choose[:, :, -1] = 1
        if not chosen:
            if print_first:
                print("\nPrinting out the maximum prob of all notes for a time step",
                      "when this maximum prob is less than 0.9")
                print_first = False
            pred_note_idx = np.argmax(y_pred_right.cpu())
            choose[:, :, pred_note_idx + 89] = 1
            if pred_note_idx != 87:  # No holds for rests
                if y_pred_right[:, pred_note_idx] - y_pred_right[:, -1] <= 0.2:  # Hold condition
                    choose[:, :, -1] = 1
            print(_, "right",
                  y_pred_right[:, np.argmax(y_pred_right.cpu())])  # Maximum probability out of all components
        x_new = torch.empty(x.shape)  # Uses the output of the last time_step
        for idx, nt in enumerate(x[1:]):  # As the input for the next time_step
            x_new[idx] = nt  # So the new sequence will be the same past sequence minus the first note
        x_new[-1] = choose
        x = x_new  # We will use this new sequence to predict in the next iteration the next note
        notes.append(choose.cpu().numpy())  # Saves the predicted note

        # Condition so that the generation does not
        # get stuck on a particular sequence
        if changing_note:
            if _ % seq_len == 0:
                if sampling_idx >= len(notes_encoded):  # TODO
                    sampling_idx = 0
                    change_note = True
                st = randint(1, 100)
                if change_note:
                    x_new[-1] = notes_encoded[sampling_idx][st, :, :]  # TODO
                    change_note = False
                else:
                    x_new[-1] = notes_encoded[sampling_idx][0, :, :]  # TODO
                sampling_idx += 1
                x = x_new

        # Condition so that the generation does not
        # get stuck on a particular note
        if _ > 6 and note_stuck:
            if (notes[-1][:, :, 89:] == notes[-2][:, :, 89:]).sum(2)[0][0].numpy() in [88, 89]:
                if (notes[-1][:, :, 89:] == notes[-3][:, :, 89:]).sum(2)[0][0].numpy() in [88, 89]:
                    if (notes[-1][:, :, 89:] == notes[-4][:, :, 89:]).sum(2)[0][0].numpy() in [88, 89]:
                        if (notes[-1][:, :, 89:] == notes[-5][:, :, 89:]).sum(2)[0][0].numpy() in [88, 89]:
                            if (notes[-1][:, :, 89:] == notes[-6][:, :, 89:]).sum(2)[0][0].numpy() in [88, 89]:
                                for m in range(5):
                                    notes.pop(-1)
                                if sampling_idx >= len(notes_encoded):  # TODO
                                    sampling_idx = 0
                                x_new[-1] = notes_encoded[sampling_idx][randint(1, 100), :, :]  # TODO
                                x = x_new
                                sampling_idx += 1

    # Gets the notes into the correct NumPy array shape
    gen_notes = np.empty((len(notes) - seq_len + 1, 178))  # Doesn't use the first predicted notes
    for idx, nt in enumerate(notes[seq_len - 1:]):  # Because these were sampled from the training data
        gen_notes[idx] = nt[0]

    # Decodes the generated music
    gen_midi_left = decode(get_tempo_dim_back(gen_notes[:, :89], 74), time_step=time_step)
    # Gets rid of too many rests
    if remove_extra_rests:
        stream_left = ms.stream.Stream()
        for idx, nt in enumerate(gen_midi_left):
            if type(nt) == ms.note.Rest and idx < len(gen_midi_left) - 5:
                if nt.duration.quarterLength > 4 * time_step:
                    print("Removing rest")
                    continue
                if type(gen_midi_left[idx + 4]) == ms.note.Rest:
                    print("Removing rest")
                    continue
                stream_left.append(nt)
            else:
                stream_left.append(nt)
    else:
        stream_left = gen_midi_left
    # Same thing for right hand
    gen_midi_right = decode(get_tempo_dim_back(gen_notes[:, 89:], 74), time_step=time_step)
    if remove_extra_rests:
        stream_right = ms.stream.Stream()
        for idx, nt in enumerate(gen_midi_right):
            if type(nt) == ms.note.Rest and idx < len(gen_midi_right) - 5:
                if nt.duration.quarterLength > 4 * time_step:
                    print("Removing rest")
                    continue
                if type(gen_midi_right[idx + 4]) == ms.note.Rest:
                    print("Removing rest")
                    continue
                stream_right.append(nt)
            else:
                stream_right.append(nt)
    else:
        stream_right = gen_midi_right

    # Saves both hands combined as a MIDI file
    combine(stream_left, stream_right, file_name + ".mid")


def ltsm_gen_v2(net, seq_len, file_name, notes_encoded, n_steps=100, hidden_size=178,
                time_step=0.05, changing_note=False, note_stuck=False, remove_extra_rests=True):

    """
    Uses the trained LSTM to generate new notes and saves the output to a MIDI file
    The difference between this and the previous one is that we only use one note as input
    And then keep generating notes until we have a sequence of notes of length = seq_len
    Once we do, we start appending the generated notes to the final output
    :param net: Trained LSTM
    :param seq_len: Length of input sequence
    :param file_name: Name to be given to the generated MIDI file
    :param n_steps: Number of vectors to generate
    :param hidden_size: Hidden size of the trained LSTM
    :param time_step: Vector duration. Should be the same as the one on get_right_hand()
    :param changing_note: To sample from different sources at some point of the generation
    and add this new note to the sequence. This is done in case the generation gets stuck
    repeating a particular sequence over and over.
    :param note_stuck: To change the note if the generation gets stuck playing the same
    note over and over.
    :param remove_extra_rests: If the generation outputs a lot of rests in between, use this
    :return: None. Just saves the generated music as a .mid file
    """

    notes = []  # Will contain a sequence of the predicted notes
    x = notes_encoded[None, :, :]  # First note of the piece
    notes.append(x.numpy())  # Saves the first note
    h_state = torch.zeros(1, 1, hidden_size).float()
    c_state = torch.zeros(1, 1, hidden_size).float()
    print_first = True
    change_note = False
    for _ in range(n_steps):
        chosen = False  # To account for when no dimension's probability is bigger than 0.9
        # breakpoint()
        net = net.double()
        y_pred, h_c_state = net(x.double(), (h_state.double(), c_state.double()))
        h_state, c_state = h_c_state[0].data, h_c_state[1].data
        y_pred = y_pred.data
        y_pred = y_pred[-1]  # We only care about the last predicted note (next note after last note of input sequence)
        choose = torch.zeros((1, 1, 178))  # Coverts the probabilities to the actual note vector
        y_pred_left = y_pred[:, :89]
        for idx in range(89):
            if y_pred_left[:, idx] > 0.9:
                choose[:, :, idx] = 1
                chosen = True
        if y_pred_left[:, -1] >= 0.7:  # We add a hold condition, in case the probability
            choose[:, :, 88] = 1  # of having a hold is close to the one of having the pitch
        if not chosen:
            if print_first:
                print("\nPrinting out the maximum prob of all notes for a time step",
                      "when this maximum prob is less than 0.9")
                print_first = False
            pred_note_idx = np.argmax(y_pred_left.cpu())
            choose[:, :, pred_note_idx] = 1
            if pred_note_idx != 87:  # No holds for rests
                if y_pred_left[:, pred_note_idx] - y_pred_left[:, -1] <= 0.2:  # Hold condition
                    choose[:, :, 88] = 1
            print(_, "left", y_pred_left[:, np.argmax(y_pred_left.cpu())])  # Maximum probability out of all components
        y_pred_right = y_pred[:, 89:]
        for idx in range(89):
            if y_pred_right[:, idx] > 0.9:
                choose[:, :, idx+89] = 1
                chosen = True
        if y_pred_right[:, -1] >= 0.7:
            choose[:, :, -1] = 1
        if not chosen:
            if print_first:
                print("\nPrinting out the maximum prob of all notes for a time step",
                      "when this maximum prob is less than 0.9")
                print_first = False
            pred_note_idx = np.argmax(y_pred_right.cpu())
            choose[:, :, pred_note_idx+89] = 1
            if pred_note_idx != 88:  # No holds for rests
                if y_pred_right[:, pred_note_idx] - y_pred_right[:, -1] <= 0.2:  # Hold condition
                    choose[:, :, -1] = 1
            print(_, "right", y_pred_right[:, np.argmax(y_pred_right.cpu())])  # Maximum probability out of all components

        # If the number of input sequences is shorter than the expected one
        if x.shape[0] < seq_len:  # We keep adding the predicted notes to this input
            x_new = torch.empty((x.shape[0] + 1, x.shape[1], x.shape[2]))
            for i in range(x_new.shape[0] - 1):
                x_new[i, :, :] = x[i, :, :]
            x_new[-1, :, :] = y_pred
            x = x_new
            notes.append(choose)
        else:  # If we already have enough sequences
            x_new = torch.empty(x.shape)  # Removes the first note
            for idx, nt in enumerate(x[1:]):  # of the current sequence
                x_new[idx] = nt  # And appends the predicted note to the
            x_new[-1] = choose  # input of sequences
            x = x_new
            notes.append(choose)

        # Condition so that the generation does not
        # get stuck on a particular sequence
        if changing_note:
            if _ % seq_len == 0:
                if sampling_idx >= len(notes_encoded):  # TODO
                    sampling_idx = 0
                    change_note = True
                st = randint(1, 100)
                if change_note:
                    x_new[-1] = notes_encoded[sampling_idx][st, :, :]  # TODO
                    change_note = False
                else:
                    x_new[-1] = notes_encoded[sampling_idx][0, :, :]  # TODO
                sampling_idx += 1
                x = x_new

        # Condition so that the generation does not
        # get stuck on a particular note
        if _ > 6 and note_stuck:
            if (notes[-1][:, :, 89:] == notes[-2][:, :, 89:]).sum(2)[0][0].numpy() in [88, 89]:
                if (notes[-1][:, :, 89:] == notes[-3][:, :, 89:]).sum(2)[0][0].numpy() in [88, 89]:
                    if (notes[-1][:, :, 89:] == notes[-4][:, :, 89:]).sum(2)[0][0].numpy() in [88, 89]:
                        if (notes[-1][:, :, 89:] == notes[-5][:, :, 89:]).sum(2)[0][0].numpy() in [88, 89]:
                            if (notes[-1][:, :, 89:] == notes[-6][:, :, 89:]).sum(2)[0][0].numpy() in [88, 89]:
                                for m in range(5):
                                    notes.pop(-1)
                                if sampling_idx >= len(notes_encoded):  # TODO
                                    sampling_idx = 0
                                x_new[-1] = notes_encoded[sampling_idx][randint(1, 100), :, :]  # TODO
                                x = x_new
                                sampling_idx += 1

    # Gets the notes into the correct NumPy array shape
    gen_notes = np.empty((len(notes)-seq_len+1, 178))  # Doesn't use the first predicted notes
    for idx, nt in enumerate(notes[seq_len-1:]):  # Because at first this will be inaccurate
        gen_notes[idx] = nt[0]

    # Decodes the generated music
    gen_midi_left = decode(get_tempo_dim_back(gen_notes[:, :89], 74), time_step=time_step)
    # Gets rid of too many rests
    if remove_extra_rests:
        stream_left = ms.stream.Stream()
        for idx, nt in enumerate(gen_midi_left):
            if type(nt) == ms.note.Rest and idx < len(gen_midi_left) - 5:
                if nt.duration.quarterLength > 4*time_step:
                    print("Removing rest")
                    continue
                if type(gen_midi_left[idx + 4]) == ms.note.Rest:
                    print("Removing rest")
                    continue
                stream_left.append(nt)
            else:
                stream_left.append(nt)
    else:
        stream_left = gen_midi_left
    # Same thing for right hand
    gen_midi_right = decode(get_tempo_dim_back(gen_notes[:, 89:], 74), time_step=time_step)
    if remove_extra_rests:
        stream_right = ms.stream.Stream()
        for idx, nt in enumerate(gen_midi_right):
            if type(nt) == ms.note.Rest and idx < len(gen_midi_right) - 5:
                if nt.duration.quarterLength > 4 * time_step:
                    print("Removing rest")
                    continue
                if type(gen_midi_right[idx + 4]) == ms.note.Rest:
                    print("Removing rest")
                    continue
                stream_right.append(nt)
            else:
                stream_right.append(nt)
    else:
        stream_right = gen_midi_right

    # Saves both hands combined as a MIDI file
    combine(stream_left, stream_right, file_name)


# -------------
# Some Attempts
# -------------


# notes_encoded = load("bach", "unknown", 1)
# net, l, ll = train_lstm_loss_whole_seq(28, lr=0.01, n_epochs=100)
# torch.save(net.state_dict(), 'lstm_whole_seq_bach_both.pkl')
# net = LSTMMusic(178, 178).cuda()
# net.load_state_dict(torch.load("lstm_whole_seq_bach_both.pkl"))
# net.eval()
# ltsm_gen_v2(net, 28, "bach_both", time_step=0.25, n_steps=400)
# The left hand offset is a bit wrong, but other than that, pretty good


# notes_encoded = load("mendelssohn", "romantic", 10)
# net, l, ll = train_lstm_loss_whole_seq(50, n_epochs=100, lr=0.01)
# torch.save(net.state_dict(), 'lstm_whole_seq_mendelssohn_both.pkl')
# net = LSTMMusic(178, 178).cuda()
# net.load_state_dict(torch.load("lstm_whole_seq_mendelssohn_both.pkl"))
# net.eval()
# ltsm_gen_v2(net, 50, "mendelssohn_both", time_step=0.25, n_steps=1000)
# Decent!

# notes_encoded = load("mozart", "sonata", 10)
# net, l, ll = train_lstm_loss_whole_seq(50, lr=0.02, n_epochs=100)
# torch.save(net.state_dict(), 'lstm_whole_seq_mozart_both.pkl')
# net = LSTMMusic(178, 178).cuda()
# net.load_state_dict(torch.load("lstm_whole_seq_mozart_both.pkl"))
# net.eval()
# ltsm_gen_v2(net, 50, "mozart_both", time_step=0.25, n_steps=400)
# Meh...

# notes_encoded = load("beethoven", "sonata", 100)
# net, l, ll = train_lstm_loss_whole_seq(50, lr=0.01, n_epochs=100)
# torch.save(net.state_dict(), 'lstm_whole_seq_beethoven_both.pkl')
# net = LSTMMusic(178, 178).cuda()
# net.load_state_dict(torch.load("lstm_whole_seq_mozart_both.pkl"))
# net.eval()
# ltsm_gen_v2(net, 50, "beethoven_both", time_step=0.25, n_steps=400)
# Pretty bad for the most part xD

