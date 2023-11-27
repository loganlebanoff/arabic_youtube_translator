import streamlit as st
import pafy
import datetime
import time
import youtube_dl
import azure.cognitiveservices.speech as speechsdk
import os
import json
import shutil
from itertools import repeat
import concurrent
#from keys import AZURE_KEY, AZURE_REGION

AZURE_KEY = st.secrets['AZURE_KEY']
AZURE_TRANSLATION_KEY = st.secrets['AZURE_TRANSLATION_KEY']
AZURE_REGION = st.secrets['AZURE_REGION']

import wave
import contextlib
import subprocess

from util import (
    download_video,
    get_video_length,
    load_speech_config,
    get_boundaries,
    process_video,
    postprocess,
    WhisperASR,
)

@st.cache(allow_output_mutation=True)
def load_asr():
    print("Loading ASR model")
    return WhisperASR()


def sleep(seconds, transcript_empty, prev_translation):
    interval = 1
    if seconds < 0:
        a=0
    while seconds >= interval:
        # if st.session_state.should_stop:
        #     transcript_empty.markdown('## stop')
        #     return
        time.sleep(interval - 0.02)
        seconds -= interval
        transcript_empty.markdown('## ' + prev_translation)
    # if st.session_state.should_stop:
    #     transcript_empty.markdown('## stop')
    #     return
    time.sleep(seconds)


def stop_current_captions():
    print('stopping')
    st.session_state.should_stop = True

def delete_cache(youtube_id):
    shutil.rmtree(os.path.join('data', youtube_id))
    st.success('Cache deleted')

def clear_url_textbox():
    st.session_state.run_id = str(int(st.session_state.run_id) + 1)

def main():
    st.session_state.should_stop = False
    if not os.path.exists('data'):
        os.makedirs('data')

    st.title('Arabic YouTube Translator')

    url_choices = [
        '',
        'https://www.youtube.com/watch?v=2VKy2VibTX0',
        'https://www.youtube.com/watch?v=9EzXFgSebKs',
        'https://www.youtube.com/watch?v=IPKe2pnzErs',
        'https://www.youtube.com/watch?v=sCEgL5bjBVM',
        'https://www.youtube.com/watch?v=AdyniMGoIjg',
    ]
    if 'run_id' not in st.session_state:
        st.session_state.run_id = '0'
    url_dropdown = st.selectbox('Select a YouTube URL:', url_choices, key=st.session_state.run_id)
    url_textbox = st.text_input('OR Copy and paste a URL here:', value='', on_change=clear_url_textbox)
    url = url_dropdown.strip() if url_dropdown != '' else url_textbox.strip()

    print(url)
    asr = load_asr()
    if url != '':
        youtube_id = url.split('?v=')[-1]
        st.sidebar.button('Delete cache for this video', on_click=delete_cache, args=[youtube_id])
        filename = os.path.join('data', youtube_id, 'full.wav')
        with st.spinner('Downloading video'):
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


        # empty.video(url, start_time=start_time)
        empty.markdown(my_html, unsafe_allow_html=True)

        speech_config = load_speech_config()
        with st.spinner('Calculating boundaries'):
            boundaries = get_boundaries(filename)
        # print(boundaries)
        # print(len(boundaries))
        with st.spinner('Processing video'):
            start_end_translation_list = process_video(speech_config, asr, youtube_id, boundaries)
        print([x[:3] for x in start_end_translation_list])
        postprocessed_start_end_translation_list = postprocess(start_end_translation_list, youtube_id)
        print(postprocessed_start_end_translation_list)
        # for clip_idx, (start, end, translation) in enumerate(start_end_translation_list):
        #     print(start, end, translation)

        st.success('Captions created!')

        is_first_running = True
        prev_translation = ''
        for clip_idx, (start, end, translation) in enumerate(postprocessed_start_end_translation_list):
            if end < start_secs:
                continue
            if clip_idx == 0:
                if start_secs == 0:
                    sleep(start, transcript_empty, prev_translation)
                elif start > start_secs:
                    sleep(start - start_secs, transcript_empty, prev_translation)
            if is_first_running:
                is_first_running = False
                sleep(2, transcript_empty, prev_translation)
            else:
                prev_end = postprocessed_start_end_translation_list[clip_idx-1][1]
                if start_secs > prev_end:
                    sleep(start - start_secs, transcript_empty, prev_translation)
                else:
                    sleep(start - prev_end, transcript_empty, prev_translation)
            if translation is None:
                translation = '.'
            prev_translation = translation
            transcript_empty.markdown('## ' + translation)
            if start_secs > start:
                sleep(end - start_secs, transcript_empty, prev_translation)
            else:
                sleep(end - start, transcript_empty, prev_translation)


if __name__ == '__main__':
    main()

