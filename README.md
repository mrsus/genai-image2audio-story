# genai-image2audio-story
Opensource LLM generates a beautiful story from the image using LangChain.

[Live Demo](https://mrsus-genai-image2audio-story.streamlit.app/)

## Steps to run locally

1. Install dependenies
``` shell
pip3 install -r requirements.txt
```
2. Run the application
``` shell
streamlit run app.py
```

## Steps to run inside docker container

1. Build the image
``` shell
docker build -t genai-image2audio .
```

2. Run the image using docker-compose
``` shell
docker-compose up -d
```
