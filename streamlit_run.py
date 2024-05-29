import streamlit as st
import functions
from datetime import datetime


def initialize_session_state():
    if 'news_data' not in st.session_state:
        st.session_state.news_data = []
        st.session_state.news_id_counter = 0
        st.session_state.init = False

    if 'comments_data' not in st.session_state:
        st.session_state.comments_data = {}


def add_comment(news_id, comment):
    if news_id in st.session_state.comments_data:
        st.session_state.comments_data[news_id].append(comment)
    else:
        st.session_state.comments_data[news_id] = [comment]


def show_comments(news_id):
    if news_id in st.session_state.comments_data:
        for comment in st.session_state.comments_data[news_id]:
            st.write(comment)


labels = ["business", "sports", "politics", "technology"]
news = [("New Business Strategy",
         "Our company has announced a new business " +
         "strategy aimed at increasing market share."),
        ("Team Wins Championship",
         "Our local sports team has won the championship" +
         "for the third consecutive year."),
        ("Government Policy Update",
         "The government has announced a new policy" +
         " initiative to address environmental issues."),
        ("New Tech Product Launch",
         "A leading technology company has launched " +
         "a new product that promises to revolutionize the industry.")]


def add_news(title, content):
    news_id = st.session_state.news_id_counter
    st.session_state.news_id_counter += 1
    st.session_state.news_data.append({
        'id': news_id,
        'title': title,
        'category': functions.predict(content, labels)[0][0],
        'content': content,
        'date': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    })


initialize_session_state()

if not st.session_state.init:
    for n in news:
        add_news(n[0], n[1])
    st.session_state.init = True

st.title("News Site")
st.header("Add News")
with st.form(key='news_form'):
    title = st.text_input("Title")
    content = st.text_area("Content")
    submit_button = st.form_submit_button(label="Add News")
    if submit_button and title and content:
        add_news(title, content)
        st.success("News Added!")

st.header("News")
selected_category = st.selectbox("Select Category to Filter", ["All"] + labels)
filtered_news = [
    news for news in st.session_state.news_data
    if selected_category == "All" or news['category'] == selected_category
]

if filtered_news:
    for news in filtered_news:
        st.subheader(news['title'])
        st.write(f"Category: {news['category']}")
        st.write(news['content'])
        st.write(f"*Date: {news['date']}*")
        st.subheader("Comments")
        show_comments(news['id'])
        st.markdown("---")
else:
    st.write("No news in this category.")

news_titles = [news['title'] for news in filtered_news]
index = st.selectbox("Select News to View",
                     range(len(news_titles)),
                     format_func=lambda i:
                         news_titles[i] if i < len(news_titles) else "")
selected_news_id = filtered_news[index]['id'] if filtered_news else None

if selected_news_id:
    st.header("Comments")
    show_comments(selected_news_id)

st.header("Add Comment")
with st.form(key='comment_form'):
    comment = st.text_area("Enter your comment")
    submit_button = st.form_submit_button(label="Add Comment")
    if submit_button and comment:
        add_comment(selected_news_id, comment)
        st.success("Comment Added!")
