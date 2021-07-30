import streamlit as st
import pafy
import datetime
import time
import youtube_dl
import azure.cognitiveservices.speech as speechsdk
import librosa
import os
import threading
# from streamlit.ReportThread import add_report_ctx
#
# # Your thread creation code:
# thread = threading.Thread(target=runInThread, args=(onExit, PopenArgs))
# add_report_ctx(thread)
# thread.start()


@st.cache
def get_video_length(filename):
    return librosa.get_duration(filename=filename)


@st.cache(hash_funcs={"builtins.SwigPyObject": lambda _: None})
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

@st.cache(hash_funcs={"builtins.SwigPyObject": lambda _: None})
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

def download_video(url, filename):
    if os.path.exists(filename):
        return
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


@st.cache
def get_boundaries(filename):
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
    return boundaries


@st.cache(hash_funcs={"builtins.SwigPyObject": lambda _: None})
def process_video(translation_config, filename, boundaries):

    from pydub import AudioSegment
    audio_segment = AudioSegment.from_wav(filename)
    start_end_translation_list = []
    for clip_idx, start, end in zip(range(len(boundaries) // 2), boundaries[::2], boundaries[1::2]):
        newAudio = audio_segment[start*1000:end*1000]
        clip_file = filename.split('.wav')[0] + '_clip_%05d.wav' % clip_idx
        newAudio.export(clip_file, format="wav")  # Exports to a wav file in the current path.
        translation = translate(translation_config, clip_file)
        start_end_translation_list.append((start, end, translation))
    return start_end_translation_list

def main():
    url = st.text_input('YouTube URL:', 'https://www.youtube.com/watch?v=2VKy2VibTX0')

    print(url)
    filename = url.split('?v=')[-1] + '.wav'
    download_video(url, filename)
    video_length = get_video_length(filename)

    empty = st.empty()

    transcript_empty = st.empty()


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

    if st.button('Stop captions'):
        st.stop()


    # empty.video(url, start_time=start_time)
    empty.markdown(my_html, unsafe_allow_html=True)

    translation_config = load_translation_config()
    boundaries = get_boundaries(filename)
    print(boundaries)
    print(len(boundaries))
    start_end_translation_list = process_video(translation_config, filename, boundaries)
    for clip_idx, (start, end, translation) in enumerate(start_end_translation_list):
        print(start, end, translation)

    is_first_running = True
    for clip_idx, (start, end, translation) in enumerate(start_end_translation_list):
        if end < start_secs:
            continue
        if clip_idx == 0:
            if start_secs == 0:
                time.sleep(start)
            elif start > start_secs:
                time.sleep(start - start_secs)
        if is_first_running:
            is_first_running = False
            time.sleep(2.5)
        else:
            prev_end = start_end_translation_list[clip_idx-1][1]
            if start_secs > prev_end:
                time.sleep(start - start_secs)
            else:
                time.sleep(start - prev_end)
        if translation is None:
            translation = '.'
        transcript_empty.markdown('## ' + translation)
        if start_secs > start:
            time.sleep(end - start_secs)
        else:
            time.sleep(end - start)


if __name__ == '__main__':
    main()

