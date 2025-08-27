from langchain.chains import ConversationalRetrievalChain
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain_community.llms import OpenAI
from langchain.memory import ConversationBufferWindowMemory
from langchain_community.callbacks import OpenAICallbackHandler
from langchain_community.document_transformers import LongContextReorder, EmbeddingsRedundantFilter
from langchain.retrievers.document_compressors import DocumentCompressorPipeline
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain.chains.query_constructor.base import AttributeInfo
from datetime import datetime
import chainlit as cl
from chainlit.input_widget import Select, Switch, Slider
import os
import constants
import openai
from langchain.prompts import (
    ChatPromptTemplate,
    PromptTemplate,
    SystemMessagePromptTemplate,
    AIMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.schema import (
    AIMessage,
    HumanMessage,
    SystemMessage
)

# Cleanup function for source strings
def string_cleanup(string):
  """A function to clean up strings in the sources from unwanted symbols"""
  return string.replace("{","").replace("}","").replace("\\","").replace("/","")

# Set OpenAI API Key
load_dotenv()
os.environ["OPENAI_API_KEY"] = constants.APIKEY
openai.api_key = constants.APIKEY 

# Load Chroma database
embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
db = Chroma(persist_directory="./vectorstore", embedding_function=embeddings)

# Set up callback handler
handler = OpenAICallbackHandler()

# Customize prompt
system_prompt_template = (
  '''You help me to extract relevant information from a case description from news items.
  The context includes extracts from relevant new items in Dutch and English.
  You help me by answering questions about the topic I wish to write a case description on.
  Yoy also help me to write parts of my case description of I ask you to do so. 
  
  If the context doesn't provide a satisfactory answer, just tell me that and don't try to make something up.
  Please try to give detailed answers and write your answers as an academic text, unless explicitly told otherwise.
  
  """
  Context: {context}
  """
  ''')

system_prompt = PromptTemplate(template=system_prompt_template,
                               input_variables=["context"])

system_message_prompt = SystemMessagePromptTemplate(prompt = system_prompt)
human_template = "{question}"
human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)
chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt, human_message_prompt])
  
# Set memory
memory = ConversationBufferWindowMemory(memory_key="chat_history", input_key='question', output_key='answer', return_messages=True, k = 5)

# Set up retriever
redundant_filter = EmbeddingsRedundantFilter(embeddings=embeddings)
reordering = LongContextReorder()
pipeline = DocumentCompressorPipeline(transformers=[redundant_filter, reordering])
retriever= ContextualCompressionRetriever(
  base_compressor=pipeline, base_retriever=db.as_retriever(search_kwargs={"k" : 20}))

# Set up source file
now = datetime.now()
timestamp = now.strftime("%Y%m%d_%H%M%S")
filename = f"answers/answers_{timestamp}.md"
with open(filename, 'w') as file:
  file.write(f"# Answers and sources for session started on {timestamp}\n\n")

@cl.on_chat_start
async def start():
  settings = await cl.ChatSettings(
    [
      Select(
        id="Model",
        label="OpenAI - Model",
        values=["gpt-4", "gpt-4-turbo", "gpt-4o-mini", "gpt-4o"],
        initial_index=1,
      ),
      Switch(id="Streaming", label="OpenAI - Stream Tokens", initial=True),
      Slider(
        id="Temperature",
        label="OpenAI - Temperature",
        initial=0,
        min=0,
        max=2,
        step=0.1,
      ),
    ]
  ).send()
  await setup_chain(settings)

@cl.on_settings_update
async def setup_chain(settings):
  # Set llm
  llm=ChatOpenAI(
    temperature=settings["Temperature"],
    streaming=settings["Streaming"],
    model=settings["Model"],
  )

  # Set up conversational chain
  chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=retriever,
    chain_type="stuff",
    return_source_documents = True,
    return_generated_question = True,
    combine_docs_chain_kwargs={'prompt': chat_prompt},
    memory=memory,
    condense_question_llm = ChatOpenAI(temperature=0, model='gpt-4-turbo'),
  )
  cl.user_session.set("chain", chain)

@cl.on_message
async def main(message: str):
  chain = cl.user_session.get("chain")
  cb = cl.LangchainCallbackHandler()
  cb.answer_reached = True
  res = await cl.make_async(chain)(message.content, callbacks=[cb])
  question = res["question"]
  answer = res["answer"]
  answer += "\n\n Sources:\n\n"
  sources = res["source_documents"]
  print_sources = []
  with open(filename, 'a') as file:
    file.write("### Query:\n")
    file.write(question)
    file.write("\n\n")
    file.write("### Answer:\n")
    file.write(res['answer'])
    file.write("\n\n")
  counter = 1
  for source in sources:
    with open(filename, 'a') as file:
      reference = os.path.basename(source.metadata['source'])
      if source.metadata['source'] not in print_sources:
        print_sources.append(source.metadata['source'])
        answer += '- '
        answer += reference
        answer += '\n'
      file.write(f"#### Document_{counter}: ")
      file.write(reference)
      file.write("\n\n")
      file.write("#### Content:\n")
      file.write(source.page_content.replace("\n", " "))
      file.write("\n\n")
      counter += 1

  await cl.Message(content=answer).send()

 
