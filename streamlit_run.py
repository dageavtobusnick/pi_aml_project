import streamlit as st
import functions
from datetime import datetime

if 'news_data' not in st.session_state:
    st.session_state.news_data = []


def add_news(title, category, content):
    st.session_state.news_data.append({
        'title': title,
        'category': functions.predict(content),
        'content': content,
        'date': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    })


st.title("Новостной сайт")

st.header("Добавить новость")
with st.form(key='news_form'):
    title = st.text_input("Заголовок")
    category = st.selectbox("Категория", ["Политика", "Экономика", "Спорт", "Технологии", "Культура", "Здоровье"])
    content = st.text_area("Содержание")
    submit_button = st.form_submit_button(label="Добавить новость")
    
    if submit_button and title and content:
        add_news(title, category, content)
        st.success("Новость добавлена!")


st.header("Новости")
selected_category = st.selectbox("Выберите категорию для фильтрации", ["Все"] + ["Политика", "Экономика", "Спорт", "Технологии", "Культура", "Здоровье"])

filtered_news = [news for news in st.session_state.news_data if selected_category == "Все" or news['category'] == selected_category]

if filtered_news:
    for news in filtered_news:
        st.subheader(news['title'])
        st.write(f"Категория: {news['category']}")
        st.write(news['content'])
        st.write(f"*Дата: {news['date']}*")
        st.markdown("---")
else:
    st.write("Нет новостей в этой категории.")