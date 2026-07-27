import os
import sys
import re
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from aiSetup import print_llm_response, get_chat_completion, get_completion_from_messages, getModerationOutput
import utils as utils
try:
    import panel as pn  # GUI
except ModuleNotFoundError as exc:
    raise ModuleNotFoundError(
        "Missing dependency 'panel'. Install with: pip install panel "
        "or pip install -r requirements.txt"
    ) from exc

pn.extension()

def process_user_message(user_input, all_messages, debug=True):
    delimiter = "```"
    
    # Step 1: Check input to see if it flags the Moderation API or is a prompt injection
    moderation_output = getModerationOutput(user_input)

    if moderation_output["flagged"]:
        print("Step 1: Input flagged by Moderation API.")
        return "Sorry, we cannot process this request."

    if debug: print("Step 1: Input passed moderation check.")
    
    category_and_product_response = utils.find_category_and_product_only(user_input, utils.get_products_and_category())
    #print(print(category_and_product_response)
    # Step 2: Extract the list of products
    category_and_product_list = utils.read_string_to_list(category_and_product_response)
    #print(category_and_product_list)

    if debug: print("Step 2: Extracted list of products.")

    # Step 3: If products are found, look them up
    product_information = utils.generate_output_string(category_and_product_list)
    if debug: print("Step 3: Looked up product information.")

    # Step 4: Answer the user question
    system_message = f"""
    You are a customer service assistant for a large electronic store. \
    Respond in a friendly and helpful tone, with concise answers. \
    Make sure to ask the user relevant follow-up questions.
    """
    messages = [
        {'role': 'system', 'content': system_message},
        {'role': 'user', 'content': f"{delimiter}{user_input}{delimiter}"},
        {'role': 'assistant', 'content': f"Relevant product information:\n{product_information}"}
    ]

    final_response = get_completion_from_messages(all_messages + messages)
    if debug:print("Step 4: Generated response to user question.")
    all_messages = all_messages + messages[1:]

    # Step 5: Put the answer through the Moderation API
    moderation_output = getModerationOutput(final_response)

    if moderation_output["flagged"]:
        if debug: print("Step 5: Response flagged by Moderation API.")
        return "Sorry, we cannot provide this information."

    if debug: print("Step 5: Response passed moderation check.")

    # Step 6: Ask the model if the response answers the initial user query well
    evaluation_system_message = """
    You are a response quality evaluator.
    Reply with only one character:
    Y = the response sufficiently answers the customer message.
    N = the response does not sufficiently answer the customer message.
    """
    user_message = f"""
    Customer message: {delimiter}{user_input}{delimiter}
    Agent response: {delimiter}{final_response}{delimiter}

    Does the response sufficiently answer the question?
    Reply with only Y or N.
    """
    messages = [
        {'role': 'system', 'content': evaluation_system_message},
        {'role': 'user', 'content': user_message}
    ]
    evaluation_response = get_completion_from_messages(messages).strip()
    if debug: print("Step 6: Model evaluated the response.")

    # Step 7: If yes, use this answer; if not, say that you will connect the user to a human
    evaluation_normalized = re.sub(r"[^a-zA-Z]", "", evaluation_response).lower()
    approved = evaluation_normalized.startswith("y") or evaluation_normalized.startswith("yes")

    if approved:
        if debug: print("Step 7: Model approved the response.")
        return final_response, all_messages
    else:
        if debug: print("Step 7: Model disapproved the response.")
        neg_str = "I'm unable to provide the information you're looking for. I'll connect you with a human representative for further assistance."
        return neg_str, all_messages

if __name__ == "__main__":
    sample_user_input = "tell me about the smartx pro phone and the fotosnap camera, the dslr one. Also what tell me about your tvs"
    sample_response, _ = process_user_message(sample_user_input, [])
    print(sample_response)

def collect_messages(debug=False):
    user_input = inp.value
    if debug: print(f"User Input = {user_input}")
    if user_input == "":
        return
    inp.value = ''
    global context
    #response, context = process_user_message(user_input, context, utils.get_products_and_category(),debug=True)
    response, context = process_user_message(user_input, context, debug=False)
    context.append({'role':'assistant', 'content':f"{response}"})
    panels.append(
        pn.Row('User:', pn.pane.Markdown(user_input, width=600)))
    panels.append(
        pn.Row('Assistant:', pn.pane.Markdown(response, width=600, styles={'background-color': '#F6F6F6'})))
 
    return pn.Column(*panels)


panels = [] # collect display 

context = [ {'role':'system', 'content':"You are Service Assistant"} ]  

inp = pn.widgets.TextInput( placeholder='Enter text here…')
button_conversation = pn.widgets.Button(name="Service Assistant")

interactive_conversation = pn.bind(collect_messages, button_conversation)

dashboard = pn.Column(
    inp,
    pn.Row(button_conversation),
    pn.panel(interactive_conversation, loading_indicator=True, height=300),
)

dashboard.servable(title="Service Assistant")

if __name__ == "__main__":
    pn.serve(dashboard, show=True, port=5006)