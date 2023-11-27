
# if __name__ == "__main__":
#     from pyannote.audio.pipelines import VoiceActivityDetection
#     pipeline = VoiceActivityDetection(
#         segmentation={"checkpoint": "pyannote/segmentation", "use_auth_token": "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"})

import time
import glob
import azure.cognitiveservices.speech as speechsdk
import streamlit as st
import os
import datetime
AZURE_KEY = st.secrets['AZURE_KEY']
AZURE_TRANSLATION_KEY = st.secrets['AZURE_TRANSLATION_KEY']

AZURE_REGION = st.secrets['AZURE_REGION']

from util import (
    download_video,
    get_video_length,
    load_speech_config,
    get_boundaries,
    process_video,
    postprocess,
    WhisperASR,
    AzureASR,
    AzureTranslator,
    GPTTranslator,
    Processor,
)

def main():
    if not os.path.exists('data'):
        os.makedirs('data')

    url = "https://www.youtube.com/watch?v=Lv5-7A6Ybq4"

    # asr = WhisperASR()
    asr = AzureASR()

    print("Loading Translator model")
    # translator = AzureTranslator()
    translator = GPTTranslator()

    processor = Processor(asr, translator)

    print(url)
    if url != '':
        youtube_id = url.split('?v=')[-1]
        filename = os.path.join('data', youtube_id, 'full.wav')
        download_video(url, filename)
        video_length = get_video_length(filename)


        time_0 = datetime.datetime(2020, 3, 16, 0, 0, 0)
        time_1 = time_0 + datetime.timedelta(seconds=video_length)
        start_time = time_0

        start_secs = int((start_time-time_0).total_seconds())
        embed_url = url.replace('watch?v=', 'embed/') + '?&autoplay=1&start=' + str(start_secs)


        boundaries = get_boundaries(filename)
        # print(boundaries)
        # print(len(boundaries))
        start_end_translation_list = process_video(processor, youtube_id, boundaries)
        print([x[:3] for x in start_end_translation_list])
        postprocessed_start_end_translation_list = postprocess(start_end_translation_list, youtube_id)
        print(postprocessed_start_end_translation_list)
        # for clip_idx, (start, end, translation) in enumerate(start_end_translation_list):
        #     print(start, end, translation)

        print(postprocessed_start_end_translation_list)


if __name__ == '__main__':
    main()








