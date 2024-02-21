# Use a pipeline as a high-level helper
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_community.llms import HuggingFaceHub
from langchain.output_parsers import ResponseSchema
from langchain.output_parsers import StructuredOutputParser
import requests
import os
import streamlit as st

HF_TOKEN = os.getenv('HUGGINGFACEHUB_API_TOKEN')
TXT2SPEECH_API_URL = "https://api-inference.huggingface.co/models/speechbrain/tts-tacotron2-ljspeech"
IMG2TXT_API_URL = "https://api-inference.huggingface.co/models/Salesforce/blip-image-captioning-base"
headers = {"Authorization": f"Bearer {HF_TOKEN}"}

def imgdata2text(data):
    response = requests.post(IMG2TXT_API_URL, headers=headers, data=data)
    return response.json()[0].get('generated_text')

def genStory(text):

    story_schema = ResponseSchema(name="story",
                                    description="Extract the value")
    
    output_parser = StructuredOutputParser.from_response_schemas([story_schema])
    format_instructions = output_parser.get_format_instructions()

    template = """
    You are an expert in story telling.
    You can generate a short story based on a single narrative, the story should be no more than 50 words.

    CONTEXT: {text}

    {format_instructions}
    """

    prompt = PromptTemplate(template=template, input_variables=['text'],format_instructions=format_instructions)


    #repo_id = 'meta-llama/Llama-2-7b-chat-hf'
    repo_id = 'mistralai/Mixtral-8x7B-Instruct-v0.1'

    llm = HuggingFaceHub(
        repo_id=repo_id, model_kwargs={"temperature": 0.2, "max_length": 100, "return_full_text": False}
        )

    story_llm_chain = LLMChain(llm=llm, prompt=prompt)
    
    story = story_llm_chain.predict(text=text, format_instructions=format_instructions)
    return output_parser.parse(story).get('story')
    

def text2speech(text):
    payload = {'inputs': text}
    response = requests.post(TXT2SPEECH_API_URL, headers=headers, json=payload)
    return response.content

def main() :
    st.set_page_config(page_title="AI Story Gen", page_icon='🤖')
    st.header('Generate an amazing story from a photo')

    uploaded_file = st.file_uploader("Browse photos", type="jpeg")

    if uploaded_file is not None:
        bytes_data = uploaded_file.getvalue()
        
        st.image(uploaded_file, 'Selected Image', use_column_width=True)

        with st.spinner('Generating image caption...'):
            caption = imgdata2text(bytes_data)

        st.subheader('Image Caption')
        st.markdown(caption)
        
        with st.spinner('Generating image story...'):
            story = genStory(caption)

        st.subheader('Image Story')
        st.markdown(story)

        with st.spinner('Generating audio...'):
            audio = text2speech(story)

        st.audio(data=audio)

if __name__ == '__main__':
    main()


#img2text('images.jpeg')
# genStory('a man carrying a woman on his back')
# text2speech(genStory(img2text('images.jpeg')))
