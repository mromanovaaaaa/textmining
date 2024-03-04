# @author: mromanovaaaaa

# streamlit run "C:\Users\mroma\OneDrive\Рабочий стол\проекты\textmining.py"
   
import streamlit as st 
import codecs #для расшифровки файлов 
import pandas as pd #для создания датафреймов
from multi_rake import Rake #библиотека для извлечения ключевых слов
from summa.summarizer import summarize #библиотека для суммаризации
from collections import Counter #для подсчета частей речи
from annotated_text import annotated_text #NER
import graphviz #визуализация синтаксического анализа
from string import punctuation #для обработки знаков препинания
#обработка русского языка
from natasha import (Segmenter, MorphVocab, NewsEmbedding, NewsMorphTagger, NewsSyntaxParser, NewsNERTagger, Doc)
#настройки библиотеки natasha
segmenter = Segmenter()
morph_vocab = MorphVocab()
emb = NewsEmbedding()
morph_tagger = NewsMorphTagger(emb)
syntax_parser = NewsSyntaxParser(emb)
ner_tagger = NewsNERTagger(emb)

#библиотека для склонения
import pymorphy2
morph = pymorphy2.MorphAnalyzer() 

#визуальные элементы приложения
st.title(':red[Text mining with python]')
st.sidebar.title('functions')
st.sidebar.info('application based on streamlit, graphzis, multi_rake, summa, pandas и natasha python libraries')

#навигация
function = st.sidebar.radio('',['key words', 'POS-tagging', 'named entity recognition',
                                'syntactic analysis', 'summarization'])

#функция для чтения файла
from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer.converter import TextConverter
from pdfminer.layout import LAParams
from pdfminer.pdfpage import PDFPage
from io import StringIO

#загрузка файлов
filetype = st.selectbox('Filetype',['', 'txt', 'pdf'])    
uploaded_file = st.file_uploader('Upload file', filetype, disabled=not bool(filetype))

#обработка pdf
def convert_pdf_to_txt_file(path):
    rsrcmgr = PDFResourceManager()
    retstr = StringIO()
    laparams = LAParams()
    device = TextConverter(rsrcmgr, retstr, laparams=laparams)
    interpreter = PDFPageInterpreter(rsrcmgr, device)
    
    for page in PDFPage.get_pages(path):
      interpreter.process_page(page)
      text = retstr.getvalue()

    device.close()
    retstr.close()
    return text

#функция для чтения файла
def readfile(file, filetype):
    if filetype == 'txt':
        bytes_data = file.getvalue()
        text = codecs.decode(bytes_data, 'utf_8_sig', 'ignore')
        text = ''.join(x for x in text if x.isprintable())
        
    if filetype == 'pdf':
        text = convert_pdf_to_txt_file(file)
    return text

#функция для согласования словосочетаний
def soglasovanie(phrase):
    words = list(morph.parse(wword)[0] for wword in phrase.split(' '))
    queue = []
    ans = ''
    nouncount = 0
    for w in words:
        if w.tag.POS == 'NOUN' or w.tag.POS == "PREP": 
            w = w.inflect({'nomn'})
            if nouncount > 0: 
                w = w.inflect({'gent'})
            form = {str(w.tag.case), str(w.tag.gender), str(w.tag.number)}
            nouncount += 1
            if queue:
                ans += ' ' + ' '.join(map(lambda x: x.inflect(form).word, queue))        
                queue = list()
            ans += ' ' + w.word
        elif w.tag.POS == 'VERB':
            if queue: 
                ans += ' ' + ' '.join(map(lambda x: x.word, queue))
            ans += ' ' + w.word
            nouncount += 1
        elif w.tag.POS == 'ADJF':
            queue.append(w)
        else:
            ans += ' ' + w.word
    return ans
        
if uploaded_file is not None:       
    text = readfile(uploaded_file, filetype)
    
if function == 'POS-tagging':
    st.header('POS-tagging')
    
    try:
        #обработка текста
        doc = Doc(text)
        doc.segment(segmenter)
        doc.tag_morph(morph_tagger)
        doc.tag_ner(ner_tagger)
            
        #сохранение информации о частях речи
        pos = [_.pos for _ in doc.tokens]
        
        #подсчет
        poscount = dict(Counter(pos))
        
        #создание датафрейма для визуализации
        df = pd.DataFrame()
        
        #universal pos-tags
        df['postags'] = ['ADJ', 'ADP', 'ADV', 'AUX', 'CCONJ', 'DET',
                         'INTJ', 'NOUN', 'NUM', 'PART', 'PRON', 'PROPN',
                         'PUNCT', 'SCONJ', 'SYM', 'VERB', 'X']
        
        #создание столбца для данных
        df['count'] = [0 for _ in range(17)]
        
        #заполнение датафрейма
        for tag, value in poscount.items():
            df.loc[df['postags'] == tag, 'count'] = value
        #визуализация
        st.bar_chart(data=df[['count', 'postags']], x = 'postags')
    except:
        'Upload file'

if function == 'named entity recognition':
    st.header('named entity recognition')
    
    try:
        #применение функции
        doc = Doc(text)
        doc.segment(segmenter)
        doc.tag_morph(morph_tagger)
        doc.tag_ner(ner_tagger)
        
        #нормализация 
        for span in doc.spans:
            span.normalize(morph_vocab)
        
        ner = []
        #визуализация
        for doctoken in doc.spans:
            ner.append((doctoken.normal, doctoken.type))
        ner = list(set(ner))
        annotated_text(ner)
    except:
        'Upload file'
        
if function == 'syntactic analysis':
    st.header('syntactic analysis')
    text = st.text_input('Write a sentence')
    
    #чтение предложения
    if text not in [None, '']:
        doc = Doc(text)
        doc.segment(segmenter)
        doc.tag_morph(morph_tagger)
        doc.parse_syntax(syntax_parser)
        
        #сохранение информации о связях между словами
        syntax = []
        for token in doc.tokens:
            syntax.append([token.text, token.id, token.head_id, token.rel])
            
        #перевод в датафрейм для более удобной индексации    
        df = pd.DataFrame(syntax)
        
        #создание графа
        graph = graphviz.Digraph()
        
        #добавление узлов и граней
        for word, x, rel in zip(df[0], df[2], df[3]):
            try:
                graph.edge(df.loc[df[1] == x, 0].values[0], word, rel)
            except Exception:
                pass
        
        #визуализация
        st.graphviz_chart(graph)

if function == 'key words':
    st.header('key words')
    try:
        if text:
            #настройка параметров функции multi-rake
            n = st.number_input('Number of key words', min_value=1, value=10, step=1)
            maxwords = st.number_input('Maximum number of keywords in phrase', min_value=1, value=3, step=1)
            minfreq = st.number_input('Minimum frequensy', min_value=1, value=2,step=1)
            lem = st.checkbox('Lemmatization (for russian language)')
                
        #лемматизация
        if lem:
            doc = Doc(text)
            doc.segment(segmenter)
            doc.tag_morph(morph_tagger)
            for token in doc.tokens:
                 token.lemmatize(morph_vocab)
            text = ' '.join([ _.lemma for _ in doc.tokens])
        
        #применение алгоритма RAKE
        rake = Rake(max_words=maxwords, min_freq=minfreq)
        keywords = rake.apply(text)
        
        if lem:
            annotated_text(*list(map(lambda x: (soglasovanie(x[0]), str(x[1])), keywords[0:n])))
        else:
            annotated_text(*list(map(lambda x: (x[0], str(x[1])), keywords[0:n])))

    except:
        'Upload file'
        
if function == 'summarization':
    st.header('summarization')
    
    try:
        #обработка исключений по длине
        summary = summarize(text, ratio=0.2, language='russian')
        ratio = 0.2
        while not summary and ratio <=1:
            ratio+=0.1
            summary = summarize(text, ratio=ratio, language='russian')
            
        #замена знаков препинания на знаки препинания с пробелом
        for punct in punctuation:
            summary = summary.replace(punct, punct + ' ', summary.count(punct)) 
        
        #вывод обобщенного текста
        st.write('---')
        st.write(summary)
        st.write('---')
        st.write('Tap to copy')
        st.code(summary)
        st.write('---')
    except:
        'Upload file'