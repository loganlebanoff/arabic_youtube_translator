import streamlit as st
import pafy

url = st.text_input('YouTube URL:', 'https://www.youtube.com/watch?v=2VKy2VibTX0')

video = pafy.new(url)
video_length = video.length

empty = st.empty()

start_time = st.slider('Time', 0, video_length, step=1)


embed_url = url.replace('watch?v=', 'embed/') + '?&autoplay=1&start=' + str(start_time)

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