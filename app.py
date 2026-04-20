import gradio as gr
import torchaudio
from processor import AudioSplitter

splitter = AudioSplitter()

def process_audio(audio_file):
    # Split audio into 4 stems
    stems, sr = splitter.split(audio_file)
    
    output_files = []
    for name, data in stems.items():
        path = f"{name}.wav"
        torchaudio.save(path, data, sr)
        output_files.append(path)
        
    return output_files

# Define a clean, basic UI
demo = gr.Interface(
    fn=process_audio,
    inputs=gr.Audio(type="filepath", label="Upload your song"),
    outputs=[
        gr.Audio(label="Drums"),
        gr.Audio(label="Bass"),
        gr.Audio(label="Instruments"),
        gr.Audio(label="Vocals")
    ],
    title="Harmonix: AI Audio Splitter",
    description="Drop any song and watch AI pull apart the vocals and instruments."
)

if __name__ == "__main__":
    demo.launch()
