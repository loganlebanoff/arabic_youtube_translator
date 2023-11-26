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

# import subprocess
# TOKEN = "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"
# subprocess.run(["huggingface-cli", "login", "--token", TOKEN])

# from pyannote.audio.core.model import Model
# Model.from_pretrained("pyannote/segmentation", use_auth_token="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")

AZURE_KEY = st.secrets['AZURE_KEY']
AZURE_TRANSLATION_KEY = st.secrets['AZURE_TRANSLATION_KEY']
AZURE_REGION = st.secrets['AZURE_REGION']

import wave
import contextlib
import subprocess

def get_video_length(filename):
    try:
        with contextlib.closing(wave.open(filename, 'r')) as f:
            frames = f.getnframes()
            rate = f.getframerate()
            duration = frames / float(rate)
    except:
        mp3_filename = filename.replace('.wav', '.mp3')
        os.rename(filename, mp3_filename)
        subprocess.call(['ffmpeg', '-i', mp3_filename, '-acodec', 'pcm_s16le', '-ac', '1', '-ar', '16000', '-vn', filename])

        with contextlib.closing(wave.open(filename, 'r')) as f:
            frames = f.getnframes()
            rate = f.getframerate()
            duration = frames / float(rate)
    return duration


def load_speech_config():
    # Set up the subscription info for the Speech Service:
    # Replace with your own subscription key and service region (e.g., "westus").
    speech_key, service_region = AZURE_KEY, AZURE_REGION
    print(speech_key, service_region)

    # Specify the path to an audio file containing speech (mono WAV / PCM with a sampling rate of 16
    # kHz).


    speech_config = speechsdk.speech.SpeechConfig(subscription=speech_key,
                                           region=service_region,
                                           speech_recognition_language='ar-EG',
    )
    speech_config.request_word_level_timestamps()
    speech_config.output_format = speechsdk.OutputFormat(1)
    return speech_config

def recognize(speech_config, file):
    audio_config = speechsdk.AudioConfig(filename=file)
    recognizer = speechsdk.SpeechRecognizer(
        speech_config=speech_config, audio_config=audio_config)

    # Starts translation, and returns after a single utterance is recognized. The end of a
    # single utterance is determined by listening for silence at the end or until a maximum of 15
    # seconds of audio is processed. The task returns the recognition text as result.
    # Note: Since recognize_once() returns only a single utterance, it is suitable only for single
    # shot recognition like command or query.
    # For long-running multi-utterance recognition, use start_continuous_recognition() instead.
    print('Recognition job started')
    result = recognizer.recognize_once()

    # Check the result
    if result.reason == speechsdk.ResultReason.RecognizedSpeech:
        print("Recognized: {}".format(result.text))
        return result.text, json.loads(result.json)
    elif result.reason == speechsdk.ResultReason.NoMatch:
        print("No speech could be recognized: {}".format(result.no_match_details))
    elif result.reason == speechsdk.ResultReason.Canceled:
        print("Translation canceled: {}".format(result.cancellation_details.reason))
        if result.cancellation_details.reason == speechsdk.CancellationReason.Error:
            print("Error details: {}".format(result.cancellation_details.error_details))
    return None, None
    a=0

def translate(recognition):
    import os, requests, uuid, json

    subscription_key = AZURE_TRANSLATION_KEY
    endpoint = 'https://api.cognitive.microsofttranslator.com'

    # If you encounter any issues with the base_url or path, make sure
    # that you are using the latest endpoint: https://docs.microsoft.com/azure/cognitive-services/translator/reference/v3-0-translate
    path = '/translate?api-version=3.0'
    params = '&from=ar-EG&to=en&includeAlignment=true'
    constructed_url = endpoint + path + params

    headers = {
        'Ocp-Apim-Subscription-Key': subscription_key,
        'Ocp-Apim-Subscription-Region': 'eastus',
        'Content-type': 'application/json',
        'X-ClientTraceId': str(uuid.uuid4()),
    }

    # You can pass more than one object in body.
    body = [{
        'text': recognition,
    }]
    print('Translation job started')
    request = requests.post(constructed_url, headers=headers, json=body)
    response = request.json()
    if 'error' in response or len(response) == 0 or len(response[0]['translations']) == 0:
        return None, None
    result = response[0]['translations'][0]
    print("Translation: {}".format(result['text']))
    return result['text'], result['alignment']['proj']

def recognize_and_translate(speech_config, file, start, end):
    recognition, recognition_info = recognize(speech_config, file)
    translation, alignment = translate(recognition)
    return recognition, recognition_info, translation, alignment, start, end

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


def get_boundaries(filename):
    boundaries_file = filename.replace('.wav', '.bounds')
    if os.path.exists(boundaries_file):
        return json.load(open(boundaries_file, encoding='utf8'))
    else:
        from pyannote.audio.pipelines import VoiceActivityDetection
        pipeline = VoiceActivityDetection(segmentation={"checkpoint": "pyannote/segmentation",  "use_auth_token": "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"})
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
        # print(vad)
        # import pdb; pdb.set_trace()
        boundaries = list(vad.get_timeline().segments_boundaries_)
        json.dump(boundaries, open(boundaries_file, 'w', encoding='utf8'))
        return boundaries


def split_clips(audio_segment, clip_file, start, end):
    if not os.path.exists(clip_file):
        newAudio = audio_segment[start*1000:end*1000]
        newAudio.export(clip_file, format="wav")  # Exports to a wav file in the current path.



def process_video(speech_config, youtube_id, boundaries):
    processed_file = os.path.join('data', youtube_id, 'processed.json')
    if os.path.exists(processed_file):
        return json.load(open(processed_file, encoding='utf8'))
    else:
        from pydub import AudioSegment
        audio_segment = AudioSegment.from_wav(os.path.join('data', youtube_id, 'full.wav'))
        start_end_translation_list = []
        clip_files = []
        starts = []
        ends = []
        for clip_idx, start, end in zip(range(len(boundaries) // 2), boundaries[::2], boundaries[1::2]):
            clip_file = os.path.join('data', youtube_id, 'clip_%05d.wav' % clip_idx)
            clip_files.append(clip_file)
            starts.append(start)
            ends.append(end)

        # We can use a with statement to ensure threads are cleaned up promptly
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
            executor.map(split_clips, repeat(audio_segment), clip_files, starts, ends)

        # We can use a with statement to ensure threads are cleaned up promptly
        # results = [recognize_and_translate(speech_config, clip_file, start, end) for clip_file, start, end in zip(clip_files, starts, ends)]
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
            results = list(executor.map(recognize_and_translate, repeat(speech_config), clip_files, starts, ends))
        for result in results:
            recognition, recognition_info, translation, alignment, start, end = result
            if translation is None:
                continue
            start_end_translation_list.append((start, end, translation, alignment, recognition, recognition_info))
        json.dump(start_end_translation_list, open(processed_file, 'w', encoding='utf8'))
        return start_end_translation_list

def parse_alignment(alignment_text):
    range_texts = alignment_text.strip().replace('-', ':').split()
    range_texts = [range_text.split(':') for range_text in range_texts]
    ranges = [(int(range_text[0]), int(range_text[1])+1, int(range_text[2]), int(range_text[3])+1) for range_text in range_texts]
    return ranges

def get_ranges_in_range(arrange_enrange, start, end):
    valid_ranges = [r for r in arrange_enrange if r[2] >= start and r[3] < end]
    return valid_ranges

def is_inside_ranges(arrange_enrange, separator_idx):
    for range in arrange_enrange:
        if separator_idx >= range[2] and separator_idx < range[3]-1:
            return True
    return False

def postprocess(start_end_translation_list, youtube_id):
    postprocessed_file = os.path.join('data', youtube_id, 'postprocessed.json')
    if os.path.exists(postprocessed_file):
        return json.load(open(postprocessed_file, encoding='utf8'))
    else:
        postprocessed = []
        for start, end, translation, alignment, recognition, recognition_info in start_end_translation_list:
            arrange_enrange = parse_alignment(alignment)
            arabic_words = recognition_info['NBest'][0]['Words']
            arabic_text = recognition
            separators = [',', '.', '?']
            arabic_separators = ['،', '.', '؟']
            separator_indices = [i for i, c in enumerate(translation) if c in separators and not is_inside_ranges(arrange_enrange, i)] + [len(translation)]


            prev_phrase_end_arabic = 0
            prev_phrase_end_english = 0
            prev_token_idx_end = -1
            prev_end_timestamp = start
            for phrase_idx, separator_idx in enumerate(separator_indices):
                english_phrase = translation[prev_phrase_end_english: separator_idx+1].strip()
                if english_phrase == '':
                    continue
                if len(english_phrase) < 15 and phrase_idx != len(separator_indices) - 1:
                    continue
                if phrase_idx != len(separator_indices) - 1:
                    next_english_phrase = translation[separator_idx+1: separator_indices[phrase_idx+1]+1].strip()
                    if len(next_english_phrase) < 15:
                        continue
                ranges = get_ranges_in_range(arrange_enrange, prev_phrase_end_english-2, separator_idx+2)
                if len(ranges) == 0:
                    continue
                max_range = max(ranges, key=lambda x: x[1])
                phrase_end = max_range[1]
                arabic_phrase = arabic_text[prev_phrase_end_arabic: phrase_end].strip()
                for sep in arabic_separators:
                    arabic_phrase = arabic_phrase.replace(sep, '')
                if arabic_phrase == '':
                    continue
                last_arabic_token = arabic_phrase.split()[-1]
                last_arabic_token_count = sum([1 for token in arabic_phrase.split() if token == last_arabic_token])
                arabic_word_matches = [(w_idx, w) for w_idx, w in enumerate(arabic_words) if w['Word'] == last_arabic_token and w_idx > prev_token_idx_end]
                if last_arabic_token_count > len(arabic_word_matches):
                    non_matches = [(w_idx, w) for w_idx, w in enumerate(arabic_words) if w['Word'] not in arabic_phrase.split() and w_idx > prev_token_idx_end]
                    if len(non_matches) == 0:
                        print(arabic_text)
                        print(translation)
                        print(english_phrase)
                        print(arrange_enrange)
                    arabic_word = non_matches[0][1]
                    arabic_word_idx = non_matches[0][0]
                    a=0
                else:
                    arabic_word = arabic_word_matches[last_arabic_token_count-1][1]
                    arabic_word_idx = arabic_word_matches[last_arabic_token_count-1][0]
                end_timestamp = start + ((arabic_word['Offset'] + arabic_word['Duration']) / 10000000)
                postprocessed.append((prev_end_timestamp, end_timestamp, english_phrase))
                prev_phrase_end_english = separator_idx+1
                prev_token_idx_end = arabic_word_idx
                prev_phrase_end_arabic = phrase_end
                prev_end_timestamp = end_timestamp
                a=0
            a=0
        return postprocessed