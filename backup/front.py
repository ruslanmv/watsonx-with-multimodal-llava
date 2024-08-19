import gradio as gr

# ... (Other functions and imports you might have)

# Create the Gradio app with custom CSS styling
with gr.Blocks(css="style.css") as demo:
    # Header
    with gr.Row(elem_id="header"):
        gr.Markdown(
            """
            # 🏨 **WatsonX Hotel Recommendation with Multimodal** 🏨
            Discover the best hotels in any city with personalized recommendations powered by WatsonX!
            """,
            elem_id="title"
        )

    # Input area for user to enter the location
    with gr.Row(elem_id="input-row"):
        place_input = gr.Textbox(
            label="Enter a Place",
            placeholder="E.g., Paris, France or Tokyo, Japan",
            lines=1,
            elem_id="place-input"
        )
        send_btn = gr.Button("Search Hotels", elem_id="search-btn")

    # Output area to show chatbot responses (including images)
    chatbot = gr.Chatbot(height=600, elem_id="chatbot-output")

    # Event: On button click, search hotels based on user input
    send_btn.click(chatbot_response, inputs=[place_input, chatbot], outputs=chatbot)

# Launch the Gradio app
demo.launch(debug=True)