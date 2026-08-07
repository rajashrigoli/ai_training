from phoenix.otel import register
from opentelemetry.trace import Status, StatusCode
import phoenix as px
import utils
import weaviate

# Kill processes on ports before importing flask_app and weaviate_server
# WARNING: Running this cell twice may kill the active kernel
utils.kill_processes_on_ports([5000, 8080, 8097, 50050, 50051])

import flask_app
import weaviate_server
import joblib
import httpx
from openai import OpenAI, DefaultHttpxClient

dash_line = "-" * 100

print("\n\n")
print(dash_line)
print("4 - Tracing and Evaluation with Weaviate")
print(dash_line)

# Clean up any existing Phoenix sessions to resolve project ID conflicts
utils.cleanup_phoenix_projects()

utils.make_url()
session = px.launch_app()

print("\n\n")
print("4.1 Configuring the tracer")
print(dash_line)

from phoenix.otel import register
import time

# Generate unique project name to avoid conflicts
phoenix_project_name = f"example-rag-pipeline-with-weaviate-{int(time.time())}"

# With phoenix, we just need to register to get the tracer provider with the appropriate endpoint. Providing auto_instrument = True, OpenAI calls are automatically traced
# TogetherAI is OpenAI compatible!
tracer_provider_phoenix = register(project_name=phoenix_project_name, endpoint="http://127.0.0.1:6006/v1/traces", auto_instrument=True)

# Retrieve a tracer for manual instrumentation
tracer = tracer_provider_phoenix.get_tracer(__name__)

print("\n\n")
print("4.2 Preparing the Weaviate client and collection")
print(dash_line)

# Connecting the weaviate client
client = weaviate.connect_to_local(port=8079, grpc_port=50050)

# Ensure FAQ collection exists with data
utils.setup_faq_collection()


data = joblib.load("faq.joblib")

# Let's recall the data structure
print(data[0])

# Loading the collection
collection = client.collections.get("Faq")
len(collection)

print("\n\n")
print("4.3 The Retriever")
print(dash_line)

def retrieve(query_text, limit=5):
    # Start a span for the query
    with tracer.start_as_current_span(
        "query_weaviate", openinference_span_kind="retriever"
    ) as span:
        # Set the input for the span
        span.set_input(query_text)

        # Query the collection
        collection_name = "Faq"
        chunks = client.collections.get(collection_name)
        results = chunks.query.near_text(query=query_text, limit=limit)

        # Set the retrieved documents as attributes on the span
        for i, document in enumerate(results.objects):
            span.set_attribute(f"retrieval.documents.{i}.document.id", str(document.uuid))
            span.set_attribute(f"retrieval.documents.{i}.document.metadata", str(document.metadata))
            span.set_attribute(
                f"retrieval.documents.{i}.document.content", str(document.properties)
            )  

        return results

# Process and format the retrieved results
@tracer.chain 
def format_context(results):
    context = ""
    for item in results.objects:
        properties = item.properties
        context += f"Question: {properties['question']}\n"
        context += f"Answer: {properties['answer']}\n"
    return context
     

# Create a prompt with the retrieved information
@tracer.chain
def create_prompt(query_text, context):
    prompt = f"""
Based on the following information, please answer the FAQ related question: "{query_text}"

Relevant FAQ (ordered by relevance):
{context}
"""
    return prompt


print("\n\n")
print("4.4 LLM call with openai library")
print(dash_line)

# Custom transport to bypass SSL verification
transport = httpx.HTTPTransport(local_address="0.0.0.0", verify=False)

# Create a DefaultHttpxClient instance with the custom transport
http_client = DefaultHttpxClient(transport=transport, headers=utils.get_proxy_headers())

# You can use any openai compatible endpoint here!
llm_client = OpenAI(
    api_key = utils.get_together_key(), # Set any as the proxy running here does not use it. Set the together api key if using the together endpoint
    base_url=utils.get_proxy_url(), # Platform-agnostic: auto-detects Coursera, Learning Platform, or Local
    http_client=http_client, # ssl bypass to make it work via proxy calls, remove it if running with together.ai endpoint 
)

# There is no need to trace as the auto_instrument was set to true
def query_openai(prompt):
    response = llm_client.chat.completions.create(
        model="Qwen/Qwen3.5-9B",
        extra_body={"reasoning": False},
        messages=[
            {"role": "system", "content": "You are a helpful assistant from a customer support."},
            {"role": "user", "content": prompt},
        ],
    )
    return response.choices[0].message.content

@tracer.chain
def rag_pipeline(query):
    # Execute the query
    retrieved_documents = retrieve(query)
    context = format_context(retrieved_documents)
    
    # Create a prompt with the retrieved information
    final_prompt = create_prompt(query, context)
    
    # Execute the OpenAI query
    final_answer = query_openai(final_prompt)

    return final_answer

response = rag_pipeline("Can I get a refund or exchange for another shirt?")
print(response)

response = rag_pipeline("What are your working hours?")
print(response)

# Checkout the traces in the Phoenix UI!
utils.make_url()