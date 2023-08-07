import streamlit as st
from espnet2.bin.tts_inference import Text2Speech
from transformers import pipeline, set_seed
import io
import soundfile as sf


def main():
    st.title("Audio Generation")
    st.write("Enter a prompt below and click 'Generate' to convert it into speech.")

    st.sidebar.subheader("TTS Model Configuration")
    st.sidebar.write("Kan Bayashi ljspeech vits")

    tts_model = Text2Speech.from_pretrained("espnet/kan-bayashi_ljspeech_vits")

    prompt = st.text_area("Enter a prompt:", "Welcome to the Bahrain Grand Prix at Bahrain International Circuit in "
                                             "Sakhir, Bahrain! The race is scheduled on 2022-03-20 at 3 PM, "
                                             "contributing to the first race of the season, 2022.We go racing today "
                                             "at Bahrain International Circuit.")

    if st.button("Generate"):
        wav = tts_model(prompt)["wav"]

        audio_bytes = io.BytesIO()
        sf.write(audio_bytes, wav.numpy(), tts_model.fs, format='wav')

        st.audio(audio_bytes, format='audio/wav')


if __name__ == "__main__":
    main()
