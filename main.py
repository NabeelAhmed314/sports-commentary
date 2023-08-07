import streamlit as st
from espnet2.bin.tts_inference import Text2Speech
from transformers import pipeline, set_seed
import io
import soundfile as sf


def main():
    st.title("Commentary Generation")
    st.write("Enter a prompt below and click 'Generate' to create commentary using GPT-2 and convert it into speech.")

    st.sidebar.subheader("TTS Model Configuration")
    st.sidebar.write("Kan Bayashi ljspeech vits")
    st.sidebar.subheader("Text Generation Model Configuration")
    st.sidebar.write("GPT-2")

    tts_model = Text2Speech.from_pretrained("espnet/kan-bayashi_ljspeech_vits")

    prompt = st.text_area("Enter a prompt:", "Welcome to the Bahrain Grand Prix at Bahrain International Circuit in "
                                             "Sakhir, Bahrain! The race is scheduled on 2022-03-20 at 3 PM, "
                                             "contributing to the first race of the season, 2022.We go racing today "
                                             "at Bahrain International Circuit.")
    num_return_sequences = 1
    max_length = st.slider("Maximum length of generated text:", min_value=10, max_value=1000, value=250)

    if st.button("Generate"):
        set_seed(42)
        generator = pipeline('text-generation', model='gpt2')
        generated_sequences = generator(prompt, max_length=max_length, num_return_sequences=num_return_sequences)

        st.write(f"{generated_sequences[0]['generated_text']}")

        wav = tts_model(generated_sequences[0]['generated_text'])["wav"]

        audio_bytes = io.BytesIO()
        sf.write(audio_bytes, wav.numpy(), tts_model.fs, format='wav')

        st.audio(audio_bytes, format='audio/wav')


if __name__ == "__main__":
    main()
