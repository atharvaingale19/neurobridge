import mne
import matplotlib.pyplot as plt
from moabb.datasets import BNCI2014_001

dataset = BNCI2014_001()
data = dataset.get_data(subjects=[1])

subject = data[1]
session = list(subject.keys())[0]
run = list(subject[session].keys())[0]

raw = subject[session][run]

print("Channels:", raw.ch_names)
print("Channel types:", raw.get_channel_types())

events, event_id = mne.events_from_annotations(raw)

print("\nEvent mapping:", event_id)

selected_events = {
    label: code
    for label, code in event_id.items()
    if label in ["left_hand", "right_hand", "feet", "tongue"]
}

epochs = mne.Epochs(
    raw,
    events,
    event_id=selected_events,
    tmin=0,
    tmax=4,
    baseline=None,
    preload=True,
    picks="eeg",
    verbose=False
)

print("\nEpoch information")
print("Shape:", epochs.get_data().shape)
print("Trials:", len(epochs))
print("Classes:", epochs.event_id)
print("Sampling frequency:", epochs.info["sfreq"])

epochs.plot(
    picks=["C3", "Cz", "C4"],
    n_epochs=3,
    n_channels=3,
    block=True
)