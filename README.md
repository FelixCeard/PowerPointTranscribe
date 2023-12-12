# PowerPointTranscribe

The idea of this project is to be able to add information said oraly as comments to the powerpoint slides. The motivation comes from the experience that often important information are disclosed during the presentation but are missing from the slide.

## Structure

Currently, the project is structured as follows:
1. Croping the slides from the video if a webcam exists
2. Detect slide changes 
3. Crop the audio to match the slide changes
4. Use speech recognition to transcribe the audio (whisper?)
5. Match the transcribed text to the slide & extract the content of the slide
6. Use an LLM (bard?) to extract the information not present in the slide 
    - Summarize the text
7. Add it back to the slides

