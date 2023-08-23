from docGPT_core import *
from RVS import *
from custom import *

set_tokens(OPENAI_TOKEN='',HF_TOKEN="")
hf_embeddings = Embedding(model_name='sentence-transformers/all-MiniLM-L6-v2', model_type = 'hf')
openai_embeddings = Embedding(model_type='openai')
persona = Persona(personality_type='explainer')
llm = Llm(model_type='gpt-3.5-turbo')

#vs = VectorStore(embedding_model=openai_embeddings, chunk_size=3000, chunk_overlap=1000)
#vs.save(store_name='store/BNF_3000_1000.pkl')
#vs = load_vectorstore('store/BNF_3000_1000.pkl')
#vs = VectorStore(embedding_model=openai_embeddings)
#chain = Chain(retriever=Retriever(vs, k=4), llm=llm, persona=persona)

import os
import gradio as gr

llm_txt = llm.model.model_name
emb_txt = openai_embeddings.model.model
status_code = "### Model specifications <br> \n" + "`LLM: " + llm_txt + "`" + " <br> `Embedding model: " + emb_txt + "`"

def chatbot(question, method):
    print(method)
    if method == 'Use Chain (Recommended)':
        result_obj = chain.qa(inputs={"query": question})
        result = result_obj['result']
        steps_text = "Running via single chain..."
    return {output: output.update(value=result),
            #acc: acc.update(value=sources)
            steps: steps.update(value=steps_text)
            }

def chatbot_sum(question, method):
    print(method)
    if method == 'Use Chain (Recommended)':
        result = keywords(vectorstore=vs,llm=llm,max_tokens=2000)
        steps_text = "Running via single chain..."
    return {output: output.update(value=result),
            #acc: acc.update(value=sources)
            steps: steps.update(value=steps_text)
            }

def get_sources_docs():
    return os.listdir('data/')

def get_vectors_list():
    directory = 'store/'
    pkl_files = [f for f in os.listdir(directory) if f.endswith('.pkl')]
    return pkl_files

def get_pkl_files():
    return ", \n".join(get_vectors_list())

def is_vectorstore_available():
    return True if len(get_vectors_list()) > 0 else False

def center_text(text):
    return "<div style='text-align:center;'>" + text + "</div>"

def set_store(store):
    global vs
    vs = load_vectorstore(store_name=f"store/{store}")
    global chain
    chain = Chain(retriever=Retriever(vs, k=3), llm=llm, persona=persona)

# def build_vs():
#     vs = VectorStore(embedding_model=openai_embeddings)
#     vs.save(store_name='store/vectorstore.pkl')
#     return {df: df.update(visible = True), btn : btn.update(visible=True), builder : builder.update(visible=False), rad:rad.update(value="Ready")}

with gr.Blocks(theme=gr.blocks.themes.Default()) as demo:
    source_list = ", \n\n".join(get_sources_docs())
    pkl_files = ", \n\n".join(get_vectors_list())

    gr.Markdown(
        """
        <h1 align="center">
        docGPT!
        </h1>
        <p align="center">
        version 1.2.
        </p>

        """
    )

    with gr.Row(equal_height=True):
        gr.Textbox(interactive=False, value=source_list, min_width=1, label="Documents in current knowledgebase")
        with gr.Column():
            v_txt = gr.Textbox(interactive=False, value=get_pkl_files(), min_width=1, label="Available vectorstores", visible=False)
            v_drp = gr.Dropdown(choices=get_vectors_list())
            v_drp.change(fn=set_store, inputs=[v_drp])
            rad = gr.Radio(choices=['Ready', 'Not built'], value= "Ready" if is_vectorstore_available() else 'Not built', interactive=False, label="Vector store status")
            stats = gr.Markdown(status_code)
            agent_toggle = gr.Radio(choices=['Use Chain (Recommended)', 'Use Agent (Experimental)'], value='Use Chain (Recommended)', label='Run method', interactive=True, visible=False)
    builder = gr.Button("Build vector store", visible = not is_vectorstore_available())


    with gr.Row(visible=True if is_vectorstore_available() else False) as df:
        input = gr.Textbox(label='Ask your question specifically.')
        output = gr.Textbox(label='Answer')
    #acc = gr.Textbox(label="Sources", max_lines=10)
    steps = gr.Textbox(label="Method info", max_lines=5)

    btn = gr.Button("Ask docGPT", visible=True if is_vectorstore_available() else False)
    btn_sum = gr.Button("Summarize", visible=True if is_vectorstore_available() else False)
    #btn.click(fn = chatbot, inputs=input, outputs=[output, acc],api_name='answer')
    btn.click(fn = chatbot, inputs=[input, agent_toggle], outputs=[output, steps],api_name='answer')
    btn_sum.click(fn = chatbot_sum, inputs=[input, agent_toggle], outputs=[output, steps],api_name='summary')

    # builder.click(fn=build_vs, outputs=[df, btn, rad, builder])

demo.launch(debug=True)
