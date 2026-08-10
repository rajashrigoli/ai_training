import os
import openai
import sys
sys.path.append('../..')

from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv()) # read local .env file
os.environ.setdefault("USER_AGENT", "langchain-doc-loader/1.0")

openai.api_key  = os.environ['GROQ_API_KEY']
dash_line = "-" * 100

print(dash_line)
print("PDF Loader Test using PyPDFLoader")
print(dash_line)

from langchain_community.document_loaders import PyPDFLoader
loader = PyPDFLoader("docs/cs229_lectures/MachineLearning-Lecture01.pdf")
pages = loader.load()

print("Length of pages:", len(pages))

page = pages[0]
print("Content:", page.page_content[0:500])
print("Metadata:", page.metadata)


print(dash_line)
print("Text Loader Test using Youtube Audio")
print(dash_line)

from langchain_community.document_loaders.generic import GenericLoader,  FileSystemBlobLoader
from langchain_community.document_loaders.parsers import OpenAIWhisperParser
from langchain_community.document_loaders.blob_loaders.youtube_audio import YoutubeAudioLoader

url="https://www.youtube.com/watch?v=jGwO_UgTS7I"
save_dir="docs/youtube/"
loader = GenericLoader(
    #YoutubeAudioLoader([url],save_dir),  # fetch from youtube
    FileSystemBlobLoader(save_dir, glob="*.m4a"),   #fetch locally
    OpenAIWhisperParser()
)
docs = loader.load()

print(docs)
#print("Content:", docs[0].page_content[0:500])

print(dash_line)
print("URLs Content Loader Test using WebBaseLoader")
print(dash_line)

from langchain_community.document_loaders import WebBaseLoader

loader = WebBaseLoader("https://github.com/basecamp/handbook/blob/master/titles-for-programmers.md")
docs = loader.load()
print("Content:", docs[0].page_content[0:500])

print(dash_line)
print("Notion Loader Test using NotionDirectoryLoader")
print(dash_line)

from langchain_community.document_loaders import NotionDirectoryLoader
loader = NotionDirectoryLoader("docs/Notion_DB")
docs = loader.load()
if docs:
    print("Content:", docs[0].page_content[0:200])
    print("Metadata:", docs[0].metadata)
else:
    print("No documents found. Ensure docs/Notion_DB contains exported Notion markdown files.")