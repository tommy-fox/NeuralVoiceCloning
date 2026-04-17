import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('/Users/tom/Downloads/Training_and_Validation_Loss_Data.csv')

# Plotting
plt.figure(figsize=(10, 6))
plt.scatter(df["Epoch"], df["Train Loss"], label="Training Loss", marker='o')
plt.scatter(df["Epoch"], df["Val Loss"], label="Validation Loss", marker='o')
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training and Validation Loss Over Epochs")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("/Users/tom/Documents/GA Tech/SP 25 Deep Learning/Final Project/NeuralVoiceCloning/SpeakerEncoding/training_loss_schedule_speaker_encoder_vctk.png")
plt.show()