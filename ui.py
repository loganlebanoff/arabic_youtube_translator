import streamlit as st
import pafy
import datetime
import time
import youtube_dl
import azure.cognitiveservices.speech as speechsdk
import os
from keys import AZURE_KEY, AZURE_REGION
import json
import wave
import contextlib

@st.cache
def get_video_length(filename):
    with contextlib.closing(wave.open(filename, 'r')) as f:
        frames = f.getnframes()
        rate = f.getframerate()
        duration = frames / float(rate)
    return duration


@st.cache(hash_funcs={"builtins.SwigPyObject": lambda _: None})
def load_translation_config():
    # Set up the subscription info for the Speech Service:
    # Replace with your own subscription key and service region (e.g., "westus").
    speech_key, service_region = AZURE_KEY, AZURE_REGION

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
        return result.translations['en'], json.loads(result.json)['Words']
    elif result.reason == speechsdk.ResultReason.RecognizedSpeech:
        print("Recognized: {}".format(result.text))
    elif result.reason == speechsdk.ResultReason.NoMatch:
        print("No speech could be recognized: {}".format(result.no_match_details))
    elif result.reason == speechsdk.ResultReason.Canceled:
        print("Translation canceled: {}".format(result.cancellation_details.reason))
        if result.cancellation_details.reason == speechsdk.CancellationReason.Error:
            print("Error details: {}".format(result.cancellation_details.error_details))
    return None, None
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
    boundaries_file = filename.replace('.wav', '.bounds')
    if os.path.exists(boundaries_file):
        return json.load(open(boundaries_file, encoding='utf8'))
    else:
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
        boundaries = list(vad._timeline.segments_boundaries_)
        json.dump(boundaries, open(boundaries_file, 'w', encoding='utf8'))
        return boundaries


@st.cache(hash_funcs={"builtins.SwigPyObject": lambda _: None})
def process_video(translation_config, youtube_id, boundaries):
    processed_file = os.path.join('data', youtube_id, 'processed.json')
    if os.path.exists(processed_file):
        return json.load(open(processed_file, encoding='utf8'))
    else:
        from pydub import AudioSegment
        audio_segment = AudioSegment.from_wav(os.path.join('data', youtube_id, 'full.wav'))
        start_end_translation_list = []
        for clip_idx, start, end in zip(range(len(boundaries) // 2), boundaries[::2], boundaries[1::2]):
            newAudio = audio_segment[start*1000:end*1000]
            clip_file = os.path.join('data', youtube_id, 'clip_%05d.wav' % clip_idx)
            newAudio.export(clip_file, format="wav")  # Exports to a wav file in the current path.
            translation, timestamps = translate(translation_config, clip_file)
            start_end_translation_list.append((start, end, translation, timestamps))
        json.dump(start_end_translation_list, open(processed_file, 'w', encoding='utf8'))
        return start_end_translation_list

@st.cache
def postprocess(start_end_translation_list, youtube_id):
    postprocessed_file = os.path.join('data', youtube_id, 'postprocessed.json')
    if os.path.exists(postprocessed_file):
        return json.load(open(postprocessed_file, encoding='utf8'))
    else:
        postprocessed = []
        for start, end, translation, timestamps in start_end_translation_list:
            tokens = translation.strip().split()
            collected_tokens = []
            for token_idx, token in enumerate(tokens):
                if token == timestamps[token_idx]:
                    collected_tokens.append(token)
                elif token in timestamps[token_idx]:
                    collected_tokens.append(token)
                    postprocessed.append(start, end, ' '.join(collected_tokens))




def sleep(seconds, transcript_empty):
    interval = 0.1
    while seconds >= interval:
        if st.session_state.should_stop:
            transcript_empty.markdown('## stop')
            return
        time.sleep(interval)
        seconds -= interval
    if st.session_state.should_stop:
        transcript_empty.markdown('## stop')
        return
    time.sleep(seconds)


def stop_current_captions():
    print('stopping')
    st.session_state.should_stop = True


def main():
    st.session_state.should_stop = False
    if not os.path.exists('data'):
        os.makedirs('data')

    url_empty = st.empty()
    url_radio = st.radio("Use example video", ['No', 'Yes'])
    initial_url = '' if url_radio == 'No' else 'https://www.youtube.com/watch?v=2VKy2VibTX0'


    url = url_empty.text_input('YouTube URL:', value=initial_url, key='youtube_url_input')

    print(url)
    if url != '':
        youtube_id = url.split('?v=')[-1]
        filename = os.path.join('data', youtube_id, 'full.wav')
        download_video(url, filename)
        video_length = get_video_length(filename)

        empty = st.empty()

        transcript_empty = st.empty()


        time_0 = datetime.datetime(2020, 3, 16, 0, 0, 0)
        time_1 = time_0 + datetime.timedelta(seconds=video_length)
        start_time = st.slider('Time', time_0, time_1, step=datetime.timedelta(seconds=1), format='HH:mm:ss', on_change=stop_current_captions)

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
        # print(boundaries)
        # print(len(boundaries))
        start_end_translation_list = process_video(translation_config, youtube_id, boundaries)
        # postprocessed_start_end_translation_list = postprocess(start_end_translation_list, youtube_id)
        # for clip_idx, (start, end, translation) in enumerate(start_end_translation_list):
        #     print(start, end, translation)

        is_first_running = True
        for clip_idx, (start, end, translation, timestamps) in enumerate(start_end_translation_list):
            if end < start_secs:
                continue
            if clip_idx == 0:
                if start_secs == 0:
                    sleep(start, transcript_empty)
                elif start > start_secs:
                    sleep(start - start_secs, transcript_empty)
            if is_first_running:
                is_first_running = False
                sleep(2.5, transcript_empty)
            else:
                prev_end = start_end_translation_list[clip_idx-1][1]
                if start_secs > prev_end:
                    sleep(start - start_secs, transcript_empty)
                else:
                    sleep(start - prev_end, transcript_empty)
            if translation is None:
                translation = '.'
            transcript_empty.markdown('## ' + translation)
            if start_secs > start:
                sleep(end - start_secs, transcript_empty)
            else:
                sleep(end - start, transcript_empty)


if __name__ == '__main__':
    main()

