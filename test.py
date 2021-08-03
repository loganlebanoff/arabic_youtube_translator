import time
import glob
import azure.cognitiveservices.speech as speechsdk
import streamlit as st
AZURE_KEY = st.secrets['AZURE_KEY']
AZURE_TRANSLATION_KEY = st.secrets['AZURE_TRANSLATION_KEY']

AZURE_REGION = st.secrets['AZURE_REGION']


def main():

    # Set up the subscription info for the Speech Service:
    # Replace with your own subscription key and service region (e.g., "westus").
    speech_key, service_region = AZURE_KEY, AZURE_REGION

    # Specify the path to an audio file containing speech (mono WAV / PCM with a sampling rate of 16
    # kHz).
    weatherfilename = "data/9EzXFgSebKs/clip_00000.wav"


    speech_config = speechsdk.SpeechConfig(subscription=speech_key,
                                           region=service_region,
                                           speech_recognition_language='ar-EG',
    )
    speech_config.request_word_level_timestamps()
    speech_config.output_format = speechsdk.OutputFormat(1)

    def recognize(file):
        audio_config = speechsdk.AudioConfig(filename=file)
        recognizer = speechsdk.SpeechRecognizer(
            speech_config=speech_config, audio_config=audio_config)

        # Starts translation, and returns after a single utterance is recognized. The end of a
        # single utterance is determined by listening for silence at the end or until a maximum of 15
        # seconds of audio is processed. The task returns the recognition text as result.
        # Note: Since recognize_once() returns only a single utterance, it is suitable only for single
        # shot recognition like command or query.
        # For long-running multi-utterance recognition, use start_continuous_recognition() instead.
        result = recognizer.recognize_once()

        # Check the result
        if result.reason == speechsdk.ResultReason.TranslatedSpeech:
            print("""Recognized: {}""".format(
                result.text))
            a=0
        elif result.reason == speechsdk.ResultReason.RecognizedSpeech:
            print("Recognized: {}".format(result.text))
        elif result.reason == speechsdk.ResultReason.NoMatch:
            print("No speech could be recognized: {}".format(result.no_match_details))
        elif result.reason == speechsdk.ResultReason.Canceled:
            print("Translation canceled: {}".format(result.cancellation_details.reason))
            if result.cancellation_details.reason == speechsdk.CancellationReason.Error:
                print("Error details: {}".format(result.cancellation_details.error_details))
        a=0
    # </TranslationOnceWithFile>
    recognize(weatherfilename)

    def translate():
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
            'text': """
            مساء النور. كنا عايزين نسألك إيه أكتر حاجة بحبها في القاهرة، القاهرة لو فضلنا نقول على الحاجات الحلوة إللي فيها بجد مش مش هنقدر اييه إنو فيها أو نديها حقها ايه بس من الحاجات دي بحبها جدا في القاهرة برج الجزيرة امم كتير القاهرة في أماكن أماكن فعلا مش هنقدر نلاقيها في أي بلد تانية والله مش عشان هي بس هي دي الحقيقة طب إيه أكتر حاجة تضايق منها خير؟

            """.strip(),
        }]
        request = requests.post(constructed_url, headers=headers, json=body)
        response = request.json()

        print(json.dumps(response, sort_keys=True, indent=4, separators=(',', ': ')))
        a=0
    translate()
    a=0


    # def speech_recognize_continuous_from_file():
    #     translation_config = speechsdk.translation.SpeechTranslationConfig(subscription=speech_key,
    #                                            region=service_region,
    #                                            speech_recognition_language='ar-EG',
    #                                            target_languages=['en-US']
    #     )
    #     translation_config.request_word_level_timestamps()
    #     translation_config.output_format = speechsdk.OutputFormat(1)
    #     audio_config = speechsdk.AudioConfig(filename=weatherfilename)
    #     recognizer = speechsdk.translation.TranslationRecognizer(
    #         translation_config=translation_config, audio_config=audio_config)
    #
    #     done = False
    #
    #
    #     def stop_cb(evt):
    #         """callback that signals to stop continuous recognition upon receiving an event `evt`"""
    #         print('CLOSING on {}'.format(evt))
    #         nonlocal done
    #         done = True
    #
    #
    #     # Connect callbacks to the events fired by the speech recognizer
    #     recognizer.recognizing.connect(lambda evt: print('RECOGNIZING: {}'.format(evt)))
    #     recognizer.recognized.connect(lambda evt: print('RECOGNIZED: {}'.format(evt)))
    #     recognizer.session_started.connect(lambda evt: print('SESSION STARTED: {}'.format(evt)))
    #     recognizer.session_stopped.connect(lambda evt: print('SESSION STOPPED {}'.format(evt)))
    #     recognizer.canceled.connect(lambda evt: print('CANCELED {}'.format(evt)))
    #     # stop continuous recognition on either session stopped or canceled events
    #     recognizer.session_stopped.connect(stop_cb)
    #     recognizer.canceled.connect(stop_cb)
    #
    #     # Start continuous speech recognition
    #     recognizer.start_continuous_recognition()
    #     while not done:
    #         time.sleep(.5)
    #
    #     recognizer.stop_continuous_recognition()
    #
    # speech_recognize_continuous_from_file()



    # import requests, uuid, json
    #
    # # Add your subscription key and endpoint
    # subscription_key = "270d94d0addb4fa5abe905da745ea56b"
    # endpoint = "https://api.cognitive.microsofttranslator.com"
    #
    # # Add your location, also known as region. The default is global.
    # # This is required if using a Cognitive Services resource.
    # location = "eastus"
    #
    # path = '/transliterate'
    # constructed_url = endpoint + path
    #
    # params = {
    #     'api-version': '3.0',
    #     'language': 'ar',
    #     'fromScript': 'arab',
    #     'toScript': 'latn'
    # }
    # constructed_url = endpoint + path
    #
    # headers = {
    #     'Ocp-Apim-Subscription-Key': subscription_key,
    #     'Ocp-Apim-Subscription-Region': location,
    #     'Content-type': 'application/json',
    #     'X-ClientTraceId': str(uuid.uuid4())
    # }
    #
    # # You can pass more than one object in body.
    # body = [{
    #     'text': 'فين'
    # }]
    #
    # request = requests.post(constructed_url, params=params, headers=headers, json=body)
    # response = request.json()
    #
    # print(json.dumps(response, sort_keys=True, indent=4, separators=(',', ': ')))

    # translate(weatherfilename)

    import youtube_dl

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
        'outtmpl': weatherfilename
    }

    with youtube_dl.YoutubeDL(ydl_opts) as ydl:
        ydl.download(['https://www.youtube.com/watch?v=2VKy2VibTX0'])
    # exit()




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
    vad = pipeline(weatherfilename)
    boundaries = vad._timeline.segments_boundaries_
    print(boundaries)

    from pydub import AudioSegment
    audio_segment = AudioSegment.from_wav(weatherfilename)
    for clip_idx, start, end in zip(range(len(boundaries) // 2), boundaries[::2], boundaries[1::2]):
        newAudio = audio_segment[start*1000:end*1000]
        clip_file = weatherfilename.split('.wav')[0] + '_clip_%05d.wav' % clip_idx
        newAudio.export(clip_file, format="wav")  # Exports to a wav file in the current path.
        translate(clip_file)


if __name__ == '__main__':
    main()










