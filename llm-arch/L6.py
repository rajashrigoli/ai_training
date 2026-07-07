#!/usr/bin/env python
# coding: utf-8

# # Lesson 6: Model Example

# In this lesson, you will reinforce your understanding of the transformer architecture by exploring the decoder-only [model](https://huggingface.co/microsoft/Phi-3-mini-4k-instruct) `microsoft/Phi-3-mini-4k-instruct`.

# ## Setup
# 
# We start with setting up the lab by installing the required libraries (`transformers` and `accelerate`) and ignoring the warnings. The `accelerate` library is required by the `Phi-3` model. But you don't need to worry about installing these libraries, the requirements for this lab are already installed. 

# <p style="background-color:#fff6ff; padding:15px; border-width:3px; border-color:#efe6ef; border-style:solid; border-radius:6px"> 💻 &nbsp; <b>Access <code>requirements.txt</code> file:</b> If you'd like to access the requirements file: 1) click on the <em>"File"</em> option on the top menu of the notebook and then 2) click on <em>"Open"</em>. For more help, please see the <em>"Appendix – Tips, Help, and Download"</em> Lesson.</p>

# In[ ]:


# !pip install transformers>=4.41.2 accelerate>=0.31.0


# In[1]:


# Warning control
import warnings
warnings.filterwarnings('ignore')


# ## Loading the LLM

# Let's first load the model and its tokenizer. For that you will first import the classes: `AutoModelForCausalLM` and `AutoTokenizer`. When you want to process a sentence, you can apply the tokenizer first and then the model in two separate steps. Or you can create a pipeline object that wraps the two steps and then apply the pipeline to the sentence. You'll explore both approaches in this notebook. This is why you'll also import the `pipeline` class.

# <p style="background-color:#fff1d7; padding:15px; "> <b>FYI: </b> The transformers library has two types of model classes: <code> AutoModelForCausalLM </code> and <code>AutoModelForMaskedLM</code>. Causal language models represent the decoder-only models that are used for text generation. They are described as causal, because to predict the next token, the model can only attend to the preceding left tokens. Masked language models represent the encoder-only models that are used for rich text representation. They are described as masked, because they are trained to predict a masked or hidden token in a sequence.</p>

# In[2]:


# import the required classes
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline


# In[3]:


# Load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("../models/microsoft/Phi-3-mini-4k-instruct")

model = AutoModelForCausalLM.from_pretrained(
    "../models/microsoft/Phi-3-mini-4k-instruct",
    device_map="cpu",
    torch_dtype="auto",
    trust_remote_code=True,
)


# <p style="background-color:#fff1d7; padding:15px; "> <b> Note:</b> You'll receive a warning that the flash-attention package is not found. That's because flash attention requires certain types of GPU hardware to run. Since the model of this lab is not using any GPU, you can ignore this warning.</p>

# Now you can wrap the model and the tokenizer in a [pipeline](https://huggingface.co/docs/transformers/en/main_classes/pipelines#transformers.pipeline) object that has "text-generation" as task.

# In[4]:


# Create a pipeline
generator = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    return_full_text=False, # False means to not include the prompt text in the returned text
    max_new_tokens=50, 
    do_sample=False, # no randomness in the generated text
)


# ## Generating a Text Response to a Prompt

# You'll now use the pipeline object (labeled as generator) to generate a response consisting of 50 tokens to the given prompt.

# <p style="background-color:#fff6e4; padding:15px; border-width:3px; border-color:#f5ecda; border-style:solid; border-radius:6px"> ⏳ <b>Note: </b> The model might take around 2 minutes to generate the output.</p>

# In[5]:


prompt = "Write an email apologizing to Sarah for the tragic gardening mishap. Explain how it happened. "

output = generator(prompt)

print(output[0]['generated_text'])


# ## Exploring the Model's Architecture

# You can print the model to take a look at its architecture.

# In[6]:


model


# The vocabulary size is 32064 tokens, and the size of the vector embedding for each token is 3072.

# In[7]:


model.model.embed_tokens


# You can just focus on printing the stack of transformer blocks without the LM head component.

# In[8]:


model.model


# There are 32 transformer blocks or layers. You can access any particular block.

# In[9]:


model.model.layers[0]


# ## Generating a Single Token to a Prompt

# You earlier used the Pipeline object to generate a text response to a prompt. The pipeline provides an abstraction to the underlying process of text generation. Each token in the text is actually generated one by one. 
# 
# Let's now give the model a prompt and check the first token it will generate.

# In[10]:


prompt = "The capital of France is"


# You'll need first to tokenize the prompt and get the ids of the tokens.

# In[11]:


# Tokenize the input prompt
input_ids = tokenizer(prompt, return_tensors="pt").input_ids
input_ids


# Let's now pass the token ids to the transformer block (before the LM head).

# In[13]:


# Get the output of the model before the lm_head
model_output = model.model(input_ids)


# The transformer block outputs for each token a vector of size 3072 (embedding size). Let's check the shape of this output.

# In[14]:


# Get the shape the output the model before the lm_head
model_output[0].shape


# The first number represents the batch size, which is 1 in this case since we have one prompt. The second number 5 represents the number of tokens. And finally 3072 represents the embedding size (the size of the vector that corresponds to each token). 
# 
# Let's now get the output of the LM head.

# In[15]:


# Get the output of the lm_head
lm_head_output = model.lm_head(model_output[0])


# In[16]:


lm_head_output.shape


# The LM head outputs for each token in the input prompt, a vector of size 32064 (vocabulary size). So there are 5 vectors, each of size 32064. Each vector can be mapped to a probability distribution, that shows the probability for each token in the vocabulary to come after the given token in the input prompt.
# 
# Since we're interested in generating the output token that comes after the last token in the input prompt ("is"), we'll focus on the last vector. So in the next cell, `lm_head_output[0,-1]` is a vector of size 32064 from which you can generate the token that comes after ("is"). You can do that by finding the id of the token that corresponds to the highest value in the vector `lm_head_output[0,-1]` (using `argmax(-1)`, -1 means across the last axis here).

# In[17]:


token_id = lm_head_output[0,-1].argmax(-1)
token_id


# Finally, let's decode the returned token id.

# In[18]:


tokenizer.decode(token_id)


# <p style="background-color:#f2f2ff; padding:15px; border-width:3px; border-color:#e2e2ff; border-style:solid; border-radius:6px"> ⬇
# &nbsp; <b>Download Notebooks:</b> If you'd like to donwload the notebook: 1) click on the <em>"File"</em> option on the top menu of the notebook and then 2) click on <em>"Download as"</em> and select <em>"Notebook (.ipynb)"</em>. For more help, please see the <em>"Appendix – Tips, Help, and Download"</em> Lesson.</p>
