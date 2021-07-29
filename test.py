import time

# try:
#     import azure.cognitiveservices.speech as speechsdk
# except ImportError:
#     print("""
#     Importing the Speech SDK for Python failed.
#     Refer to
#     https://docs.microsoft.com/azure/cognitive-services/speech-service/quickstart-python for
#     installation instructions.
#     """)
#     import sys
#     sys.exit(1)
#
# # Set up the subscription info for the Speech Service:
# # Replace with your own subscription key and service region (e.g., "westus").
# speech_key, service_region = "0340be178e1e4695bdd5540353f6a949", "eastus"
#
# # Specify the path to an audio file containing speech (mono WAV / PCM with a sampling rate of 16
# # kHz).
# weatherfilename = "learn.wav"
#
#
# translation_config = speechsdk.translation.SpeechTranslationConfig(subscription=speech_key,
#                                        region=service_region,
#                                        speech_recognition_language='ar-EG',
#                                        target_languages=['en-US']
# )
# translation_config.request_word_level_timestamps()
# translation_config.output_format = speechsdk.OutputFormat(1)
# audio_config = speechsdk.AudioConfig(filename=weatherfilename)
# recognizer = speechsdk.translation.TranslationRecognizer(
#     translation_config=translation_config, audio_config=audio_config)
#
# # Starts translation, and returns after a single utterance is recognized. The end of a
# # single utterance is determined by listening for silence at the end or until a maximum of 15
# # seconds of audio is processed. The task returns the recognition text as result.
# # Note: Since recognize_once() returns only a single utterance, it is suitable only for single
# # shot recognition like command or query.
# # For long-running multi-utterance recognition, use start_continuous_recognition() instead.
# result = recognizer.recognize_once()
#
# # Check the result
# if result.reason == speechsdk.ResultReason.TranslatedSpeech:
#     print("""Recognized: {}
#     English translation: {}""".format(
#         result.text, result.translations['en']))
#     a=0
# elif result.reason == speechsdk.ResultReason.RecognizedSpeech:
#     print("Recognized: {}".format(result.text))
# elif result.reason == speechsdk.ResultReason.NoMatch:
#     print("No speech could be recognized: {}".format(result.no_match_details))
# elif result.reason == speechsdk.ResultReason.Canceled:
#     print("Translation canceled: {}".format(result.cancellation_details.reason))
#     if result.cancellation_details.reason == speechsdk.CancellationReason.Error:
#         print("Error details: {}".format(result.cancellation_details.error_details))
# a=0
# # </TranslationOnceWithFile>



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


def main():
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
    vad = pipeline("learn.wav")
    print(vad._timeline.segments_boundaries_)

if __name__ == '__main__':
    main()










