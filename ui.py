import streamlit as st
import pafy
import datetime

url = st.text_input('YouTube URL:', 'https://www.youtube.com/watch?v=2VKy2VibTX0')

video = pafy.new(url)
video_length = video.length

empty = st.empty()


time_0 = datetime.datetime(2020, 3, 16, 0, 0, 0)
time_1 = time_0 + datetime.timedelta(seconds=video.length)
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