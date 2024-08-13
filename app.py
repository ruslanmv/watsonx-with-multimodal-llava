import gradio as gr
from gradio_multimodalchatbot import MultimodalChatbot
from gradio.data_classes import FileData
from backend import search_hotel

def get_hotel_recommendations(place):
    description_df = search_hotel(place)
    
    conversation = []
    for _, row in description_df.iterrows():
        hotel_name = row['hotel_name']
        description = row['description']
        img = row['image']

        # Save image temporarily to send as a file in chat
        img_path = f"{hotel_name}.png"
        img.save(img_path)

        user_msg = {"text": f"I want to see hotels in {place}.", "files": []}
        bot_msg = {
            "text": f"Here is {hotel_name}. {description}",
            "files": [{"file": FileData(path=img_path)}]
        }
        conversation.append([user_msg, bot_msg])

    return conversation

def chatbot_response(user_input):
    return get_hotel_recommendations(user_input)

with gr.Blocks() as demo:
    conversation = []  # Initial empty conversation
    chatbot = MultimodalChatbot(value=conversation, height=800)
    
    with gr.Row():
        place_input = gr.Textbox(label="Enter a place")
        send_btn = gr.Button("Send")

    send_btn.click(chatbot_response, inputs=place_input, outputs=chatbot)

demo.launch()
