import streamlit as st
st.set_page_config(page_title='Test', layout='wide')
import os
ROOT_DIR = os.path.dirname(os.path.abspath(__file__)) # This is your Project Root
from spacy import displacy
import pandas as pd
from transformers import T5ForConditionalGeneration, AutoTokenizer
from transformers import pipeline
from langchain import PromptTemplate, LLMChain
from langchain.llms import HuggingFacePipeline
import torch
#os.environ["CUDA_VISIBLE_DEVICES"] = "0"


if 'button_sent' not in st.session_state:
    st.session_state.button_sent = False
if 'tokenizer' not in st.session_state:
    st.session_state.tokenizer = None
if 'model' not in st.session_state:
    st.session_state.model = None
if 'model_name' not in st.session_state:
    st.session_state.model_name = None



# st.sidebar.title('How to use')
# st.sidebar.write('1. Select LLM')
# st.sidebar.write('2. Select entities/toxicity types to be used')
# st.sidebar.write('3. Enter Input and get predictions 😃')
option = st.sidebar.selectbox(
    'Select Entities',
    (['Coarse-grained', 'Fine-grained-Person', 'Fine-grained-Organization', 'Fine-grained-Location']))

if option =='Coarse-grained':
    ent_template = """In this task, you will be presented with a question in English language, and you have to write the named entities from the question if present. B denotes the first item of a phrase and an I any non-initial word. Here is the list of terms used: person names (PER), organizations (ORG), and locations (LOC). There can be instances with no named entities, then return 'None'.\n\nInput: """
if option == 'Fine-grained-Organization':
    ent_template = """In this task, you will be presented with a question in English language, and you have to write the organization named entities from the question if present. B denotes the first item of a phrase and an I any non-initial word. Here is the list of terms used: education (EDU), company (COM), religion (REL), and politicalparty names (POL). There can be instances with no named entities, then return 'None'.\n\nInput: """
if option == 'Fine-grained-Person':
    ent_template = """In this task, you will be presented with a question in English language, and you have to write the person named entities from the question if present. B denotes the first item of a phrase and an I any non-initial word. Here is the list of terms used: actor (ACT), athlete (ATH), soldier (SOL), and politician names (POL). There can be instances with no named entities, then return 'None'.\n\nInput: """
if option == 'Fine-grained-Location':
    ent_template = """In this task, you will be presented with a question in English language, and you have to write the location named entities from the question if present. B denotes the first item of a phrase and an I any non-initial word. Here is the list of terms used: city (CIT), country (COU), and state names (STA). There can be instances with no named entities, then return 'None'.\n\nInput: """
#     if st.session_state.model_name is None or st.session_state.model_name != option:
# st.session_state.model_name = 'google/flan-ul2'
# st.session_state.tokenizer = AutoTokenizer.from_pretrained('google/flan-ul2')

if 'google/flan-ul2':
    if st.session_state.model_name is None or st.session_state.model_name != 'google/flan-ul2':
        st.session_state.model_name = 'google/flan-ul2'
        st.session_state.tokenizer = AutoTokenizer.from_pretrained('google/flan-ul2', model_max_length=400)
        st.session_state.model = T5ForConditionalGeneration.from_pretrained('google/flan-ul2', max_length = 400, torch_dtype=torch.bfloat16)#.to(torch.device('cuda:0'))
        st.session_state.model.cuda(0)
tab1, tab2 = st.tabs(["Entity Recognition", "Toxicity Detection"])

def get_ent_dict(dat, ents):
    ret_dict = {
        "text": dat,
        "title": ""
    }
    entss = []
    for i in ents:
        if dat.find(i[0]) != -1:
            entss.append({"start": dat.find(i[0]), "end": dat.find(i[0])+len(i[0]), "label": i[1]})
    ret_dict['ents'] = entss
    return ret_dict

def get_full_ents(da):
    entities = []
    term = []
    typ = ''
    for i in da:
        if len(i) == 1:
            continue
        if 'B-' in i[1]:
            if term:
                entities.append((' '.join(term), typ))
                term = []
                typ = ''
            term.append(i[0])
            typ = i[1].split('-')[1]
        elif 'I-' in i[1]:
            term.append(i[0])
            typ = i[1].split('-')[1]
    if term:
        entities.append((' '.join(term), typ))
        term = []
        typ = ''
    return entities

def predict_entities(inttt):
    input_text = inttt
    print(input_text)
    print(st.session_state.tokenizer)
    input_ids = st.session_state.tokenizer(input_text, return_tensors="pt").input_ids.to(0)

    outputs = st.session_state.model.generate(input_ids, max_length=400, do_sample=False)
    sent = st.session_state.tokenizer.decode(outputs[0])
    sent = sent.replace('<pad>', '').replace('</s>', '').strip()
    #result = st.session_state.llm_chain.predict(instruction=input_text)
    #print(result)
    return sent


with tab1:
    st.title('Zero Shot Entity Recognition')
    #ent_template = """In this task, you will be presented with a question in English language, and you have to write the named entities from the question if present. B denotes the first item of a phrase and an I any non-initial word. Here is the list of terms used: person names (PER), organizations (ORG), locations (LOC) and miscellaneous names (MISC). There can be instances with no named entities, then return 'None'.\n\nInput: """
    text_inp = st.text_area('Enter your input')

    # options = st.multiselect(
    #     'Select Entities',
    #     ['person names (PER)', 'organizations (ORG)', 'locations (LOC)', 'miscellaneous names (MISC)'],
    #     ['person names (PER)', 'organizations (ORG)', 'locations (LOC)', 'miscellaneous names (MISC)'])

    # custom_ents = st.text_input('Custom Entities:  Format :  <ent_name1> (ent_code1),<ent_name2> (ent_code2), ...  E.g.  Soldier (SOLx)')

    btn_ner = st.button('Submit')
    if btn_ner:
    #     if options or custom_ents.strip() != '':
    #         values = 
    #         if custom_ents.strip() != '':
    #             options += custom_ents.split(',')
    #         if len(options) == 1:
    #             ent_val = ' '.join(options)
    #         else:
    #             ent_val = ' '.join(options[:-1]) + ' and ' + options[-1]

        txt = text_inp.split('\n')
        st.subheader('Output')
        for t in txt:
            innnn = ent_template+t
            pred = predict_entities(innnn)
            pred = [[k.strip() for k in p.split(':')] for p in pred.split(',')]
            # print(pred)

            dic_ents = get_ent_dict(t, get_full_ents(pred))
            # if dic_ents['ents']:
            print(dic_ents['ents'])
            ent_html = displacy.render(dic_ents, manual=True, style="ent")
            print(ent_html)
            # time.sleep(1)
            st.markdown(ent_html, unsafe_allow_html=True)


with tab2:
    td_template = 'Categorize the comment on the basis of toxicity. If the comment is toxic (harmful, malicious, derogatory, threat, insult, identity attack, etc.) output Yes, otherwise output No.\n\nInput: '
    st.title('Zero Shot Toxicity Detection')

    text_inp1 = st.text_area('Enter your input ')

    # options1 = st.multiselect(
    #     'Toxicity Types',
    #     ['harmful', 'malicious', 'derogatory', 'threat', 'insult', 'identity attack', 'sexual comments'],
    # ['harmful', 'malicious', 'derogatory', 'threat', 'insult', 'identity attack', 'sexual comments'])

    btn_td = st.button('Submit ')
    if btn_td:
        st.subheader('Prediction')
        # if options1:
        #     tds = ', '.join(options1)

        txt = text_inp1.split('\n')
        data = []
        for t in txt:
            out = predict_entities(td_template+t)
            data.append((t, out.strip().lower()))
            # if out.strip().lower() == 'yes':
            #     st.success('Yes')
            # else:
            #     st.error('No')
        df = pd.DataFrame(data, columns=['Text', 'Toxicity Present?'])
        st.table(df)