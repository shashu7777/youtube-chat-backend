from flask import Flask, request, jsonify
from flask_cors import CORS
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain.schema.runnable import RunnableParallel, RunnableLambda, RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser
import os
from dotenv import load_dotenv

app = Flask(__name__)
CORS(app)
load_dotenv()

api_key = os.getenv("GOOGLE_API_KEY")

@app.route("/query", methods=["POST"])
def query():
    data = request.json
    query = data.get("query")
    video_id = data.get("videoId")

    try:
      proxies = {
       'http': 'http://159.203.61.169:3128',
       'https': 'http://159.203.61.169:3128'
      }
      transcript_list=YouTubeTranscriptApi.get_transcript(video_id=video_id,languages=['en','hi'], proxies=proxies)
      transcript=""
      for chunk in transcript_list:
          transcript=transcript+chunk['text']

    except TranscriptsDisabled:
          print("No captions available for this video.")
           

    splitter=RecursiveCharacterTextSplitter(chunk_size=1000,
                                        chunk_overlap=200)

    docs=splitter.create_documents([transcript])      
    


    embedding=GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store=FAISS.from_documents(docs,embedding)

    retriever=vector_store.as_retriever(search_type="similarity",kwargs={'k':4})

    def format_docs(docs):
        context_text = "\n\n".join(doc.page_content for doc in docs)
        return context_text
    


    llm=ChatGoogleGenerativeAI(
       model='gemini-1.5-flash',
       temperature=0.2,
       google_api_key=api_key
       )

    prompt=PromptTemplate(
       input_variables=['context','query'],
       template='''
     You are a helpful assistant.
      Answer ONLY from the provided transcript context.
      If the context is insufficient, just say you don't know.

      {context}
      Question: {query}
    '''
    )

    parser=StrOutputParser()
    runnable_parallel=RunnableParallel(
       {
        'context':retriever | RunnableLambda(format_docs),
        'query':RunnablePassthrough()
       }
    )

    chain=runnable_parallel | prompt | llm | parser
    result=chain.invoke(query)

    return jsonify({"answer": result})


@app.get('/hello')
def hello():
    print("hi")
    return jsonify({'answer':'hello jdkskdskksmmsd kmdmsmdksmkkmdlmddlsdmksmdkmsd msdkmwdmkdmkwd kmdmwmdkwmd'})



if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
