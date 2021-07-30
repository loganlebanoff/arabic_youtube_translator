import streamlit as st
import pafy
import datetime
import time
import youtube_dl
import azure.cognitiveservices.speech as speechsdk
import librosa


# # @st.cache
def get_video_length(filename):
    return librosa.get_duration(filename=filename)


# @st.cache
def load_translation_config():
    # Set up the subscription info for the Speech Service:
    # Replace with your own subscription key and service region (e.g., "westus").
    speech_key, service_region = "0340be178e1e4695bdd5540353f6a949", "eastus"

    # Specify the path to an audio file containing speech (mono WAV / PCM with a sampling rate of 16
    # kHz).


    translation_config = speechsdk.translation.SpeechTranslationConfig(subscription=speech_key,
                                           region=service_region,
                                           speech_recognition_language='ar-EG',
                                           target_languages=['en-US']
    )
    translation_config.request_word_level_timestamps()
    translation_config.output_format = speechsdk.OutputFormat(1)
    return translation_config

# @st.cache
def translate(translation_config, file):
    audio_config = speechsdk.AudioConfig(filename=file)
    recognizer = speechsdk.translation.TranslationRecognizer(
        translation_config=translation_config, audio_config=audio_config)

    # Starts translation, and returns after a single utterance is recognized. The end of a
    # single utterance is determined by listening for silence at the end or until a maximum of 15
    # seconds of audio is processed. The task returns the recognition text as result.
    # Note: Since recognize_once() returns only a single utterance, it is suitable only for single
    # shot recognition like command or query.
    # For long-running multi-utterance recognition, use start_continuous_recognition() instead.
    result = recognizer.recognize_once()

    # Check the result
    if result.reason == speechsdk.ResultReason.TranslatedSpeech:
        print("""Recognized: {}
        English translation: {}""".format(
            result.text, result.translations['en']))
        a=0
        return result.translations['en']
    elif result.reason == speechsdk.ResultReason.RecognizedSpeech:
        print("Recognized: {}".format(result.text))
    elif result.reason == speechsdk.ResultReason.NoMatch:
        print("No speech could be recognized: {}".format(result.no_match_details))
    elif result.reason == speechsdk.ResultReason.Canceled:
        print("Translation canceled: {}".format(result.cancellation_details.reason))
        if result.cancellation_details.reason == speechsdk.CancellationReason.Error:
            print("Error details: {}".format(result.cancellation_details.error_details))
    a=0

# @st.cache
def download_video(url, filename):
    ydl_opts = {
        'format': 'bestaudio/best',
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'wav',
            # 'preferredquality': '192'
        }],
        'postprocessor_args': [
            '-f', 'wav',
            '-ar', '16000',
            '-acodec', 'pcm_s16le',
            '-ac', '1',
            '-vn'
        ],
        'prefer_ffmpeg': True,
        'keepvideo': False,
        'outtmpl': filename
    }

    with youtube_dl.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])
    # exit()



# @st.cache
def process_video(translation_config, filename, transcript_empty):
    from pyannote.audio.pipelines import VoiceActivityDetection
    pipeline = VoiceActivityDetection(segmentation="pyannote/segmentation")
    HYPER_PARAMETERS = {
      # onset/offset activation thresholds
      "onset": 0.8, "offset": 0.8,
      # remove speech regions shorter than that many seconds.
      "min_duration_on": 0.0,
      # fill non-speech regions shorter than that many seconds.
      "min_duration_off": 0.0
    }
    pipeline.instantiate(HYPER_PARAMETERS)
    vad = pipeline(filename)
    boundaries = vad._timeline.segments_boundaries_
    print(boundaries)

    from pydub import AudioSegment
    audio_segment = AudioSegment.from_wav(filename)
    for clip_idx, start, end in zip(range(len(boundaries) // 2), boundaries[::2], boundaries[1::2]):
        newAudio = audio_segment[start*1000:end*1000]
        clip_file = filename.split('.wav')[0] + '_clip_%05d.wav' % clip_idx
        newAudio.export(clip_file, format="wav")  # Exports to a wav file in the current path.
        translation = translate(translation_config, clip_file)
        transcript_empty.markdown(translation)

def main():
    url = st.text_input('YouTube URL:', 'https://www.youtube.com/watch?v=2VKy2VibTX0')

    print(url)
    filename = url.split('?v=')[-1] + '.wav'
    download_video(url, filename)
    video_length = get_video_length(filename)

    empty = st.empty()


    time_0 = datetime.datetime(2020, 3, 16, 0, 0, 0)
    time_1 = time_0 + datetime.timedelta(seconds=video_length)
    start_time = st.slider('Time', time_0, time_1, step=datetime.timedelta(seconds=1), format='HH:mm:ss')

    start_secs = int((start_time-time_0).total_seconds())
    embed_url = url.replace('watch?v=', 'embed/') + '?&autoplay=1&start=' + str(start_secs)

    my_html = '''<style>
.video-background { 
  position: relative;
  padding-bottom: 56.25%;
  /* 16:9 */
  padding-top: 25px;
  height: 0;
}

.video-background iframe {
  position: absolute;
  top: 0;
  left: 0;
  width: 100%;
  height: 100%;
}
</style>
<div class="video-background">
<iframe src="<video>" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>
</div>
    '''.replace('<video>', embed_url)


    # empty.video(url, start_time=start_time)
    empty.markdown(my_html, unsafe_allow_html=True)

    transcript_empty = st.empty()

    translation_config = load_translation_config()
    process_video(translation_config, filename, transcript_empty)


if __name__ == '__main__':
    main()

