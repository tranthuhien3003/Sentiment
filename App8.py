import nltk
nltk.download('punkt')
nltk.download('stopwords')

import streamlit as st
import pandas as pd
import joblib
from collections import Counter
from function import extract_city, fast_translate, process_text
import datetime
import os
import joblib
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from collections import Counter
from PIL import Image

#===========================================
if "recent_reviews" not in st.session_state:
    st.session_state["recent_reviews"] = []
# ================================
# Load dữ liệu và model
# ================================
Companies = pd.read_excel("Companies_Clean.xlsx")
Reviews = pd.read_excel("Reviews_Cluster.xlsx")


positive_words = [
    "good", "great", "excellent", "efficient", "supportive", "friendly", "creative",
    "enthusiastic", "passionate", "dedicated", "professional", "reliable", "fun",
    "motivated", "inspiring", "productive", "collaborative", "trustworthy", "cheerful",
    "positive", "comfortable", "encouraging", "flexible", "respectful", "engaging",
    "helpful", "innovative", "stable", "welcoming", "rewarding"
]


negative_words = [
    "bad", "poor", "inefficient", "stressful", "toxic", "unfriendly", "boring",
    "unmotivated", "disorganized", "unprofessional", "rude", "inflexible", "overworked",
    "underpaid", "frustrating", "micromanaged", "unfair", "slow", "confusing",
    "demanding", "negative", "pressured", "annoying", "hostile", "exhausting",
    "chaotic", "inconsistent", "isolated", "unappreciated", "low"
]


def count_pos_neg_words(text, pos_words, neg_words):
    text = text.replace("_", " ").lower()
    tokens = text.split()
    counter = Counter(tokens)
    pos_count = sum(counter[w] for w in pos_words if w in counter)
    neg_count = sum(counter[w] for w in neg_words if w in counter)
    return pos_count, neg_count

# ================================
# Session state để điều hướng tab
# ================================
if 'active_tab' not in st.session_state:
    st.session_state.active_tab = "Home"

def switch_tab(tab_name):
    st.session_state.active_tab = tab_name

# ================================
# Thanh sidebar menu
# ================================
# Set default tab nếu chưa có
if "active_tab" not in st.session_state:
    st.session_state.active_tab = "Home"

# CSS cho sidebar button
st.markdown("""
    <style>
    .stButton > button {
        width: 100%;
        padding: 0.6rem 1rem;
        margin-bottom: 0.5rem;
        background-color: white;
        border: 1px solid #ccc;
        border-radius: 8px;
        text-align: left;
        font-size: 16px;
        color: black !important;
        font-weight: 500;
        transition: all 0.2s ease;
    }
    .stButton > button:hover {
        background-color: #f0f2f6;
        border-color: #888;
        color: black !important;
    }
    </style>
""", unsafe_allow_html=True)

# Sidebar navigation
with st.sidebar:
    st.markdown("### 📁 Navigation")

    if st.button("🏠 Home"):
        st.session_state.active_tab = "Home"
    if st.button("📊 Dashboard"):
        st.session_state.active_tab = "Dashboard"
    if st.button("💬 Sentiment & Clustering"):
        st.session_state.active_tab = "Sentiment"


# ================================
# Trang chủ (Home)
# ================================
if st.session_state.active_tab == "Home":
    st.markdown("""
        <div style="
            background-color: #f5f5f5; 
            padding: 4px; 
            border-radius: 8px; 
            text-align: center; 
            max-width: 1000px; 
            margin: auto;
        ">
            <h1 style="color: #333333; margin: 0;">Employee Sentiment Analysis & Review Clustering</h1>
        </div>
    """, unsafe_allow_html=True)

    


    # ✅ Hiển thị hình ảnh ở đầu trang
    from PIL import Image
    image_path = "sentiment_Cover.jpg"
    image = Image.open(image_path)
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.image(image, width=400)

    st.markdown("---")

    #📌About Online Recruitment Website
    st.subheader("💡 About Online Recruitment Website")
    st.markdown("""
    This project is based on reviews collected from **one of Vietnam’s leading online recruitment platforms** in the IT sector.  
    The platform connects tech companies with skilled IT professionals and developers across Vietnam.  
    📝 **Note**: The data used here is **dummy data** and is intended for **academic and research purposes only**.
    """)

    # 🎯 Project Scope
    st.subheader("🎯 Project Scope")
    st.markdown("""
    This study uses employee review data to provide useful insights to companies.  
    Using **Natural Language Processing (NLP)**, the system helps organizations understand:
    - 😊 How satisfied employees are  
    - ✅ What works well and what needs improvement  
    - 👀 How the company appears to IT job seekers
    """)


# ================================
# Tab 2: Dashboard
# ================================
elif st.session_state.active_tab == "Dashboard":
    st.markdown("""
        <div style="
            background-color: #f5f5f5; 
            padding: 4px; 
            border-radius: 8px; 
            text-align: center; 
            max-width: 800px; 
            margin: auto;
        ">
            <h1 style="color: #333333; margin: 0;">Sentiment and Clustering Dashboard</h1>
        </div>
    """, unsafe_allow_html=True)

    # Tạo cột thành phố từ Location
    Companies["City"] = Companies["Location"].apply(extract_city)

    # Thống kê tổng quan
    st.subheader("🏢 Data Overview")
    st.markdown(f"**Total companies**: {Companies['Company Name'].nunique()}")
    st.markdown(f"**Total reviews**: {Reviews.shape[0]}")

    # -------------------------------
    # 📊 Biểu đồ phân phối cảm xúc
    # -------------------------------
    st.subheader("📊 Sentiment Dashboard")
    with st.expander("📂 Click to see Insights", expanded=False):

        col1, col2 = st.columns(2)
        with col1:
            sentiment_count = Reviews["label_sentiment"].value_counts().sort_index()
            sentiment_labels = {0: "😠 Negative", 1: "😐 Neutral", 2: "😊 Positive"}
            sentiment_df = pd.DataFrame({
                "Sentiment": [sentiment_labels[i] for i in sentiment_count.index],
                "Count": sentiment_count.values
            })

            fig = px.bar(
                sentiment_df,
                x="Sentiment",
                y="Count",
                color_discrete_sequence=["#4F81BD"]
            )
            fig.update_layout(
                title="Sentiment Distribution",
                xaxis_title=None,
                yaxis_title="Number of Reviews",
                showlegend=False,
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.markdown("✔️ The data comes from over **{} reviews** across **{} companies**.".format(
                Reviews.shape[0],
                Reviews["Company Name"].nunique()
            ))

        # -------------------------------
        # 📈 Biểu đồ cảm xúc trung bình
        # -------------------------------
        st.subheader("📈 Average Sentiment by Company Attributes")

        # Gán Location từ Companies nếu chưa có
        if "Location" not in Reviews.columns:
            Reviews = Reviews.merge(Companies[["id", "Location"]], on="id", how="left")

        # Tạo City từ Location
        Reviews["City"] = Reviews["Location"].apply(extract_city)

        # Các cột cần phân tích
        group_columns = ["Company Type", "Company industry", "Company size", "City"]

        for col in group_columns:
            st.markdown(f"### 🔹 **{col}**")

            # Tính cả số lượng reviews cho từng nhóm
            grouped = Reviews.groupby(col)["label_sentiment"].agg(["mean", "count"]).sort_values(by="mean", ascending=False).head(10).reset_index()
            grouped.columns = [col, "Average Sentiment", "Review Count"]

            fig = px.bar(
                grouped,
                x="Average Sentiment",
                y=col,
                orientation='h',
                color="Average Sentiment",
                color_continuous_scale=[[0, "#C0504D"], [0.5, "#F79646"], [1.0, "#4F81BD"]],
                hover_data={col: True, "Average Sentiment": True, "Review Count": True},
            )
            fig.update_layout(
                xaxis_title="Average Sentiment Score (0 = Negative, 2 = Positive)",
                yaxis_title=None,
                showlegend=False,
                height=400
            )

            st.plotly_chart(fig, use_container_width=True)

        # -------------------------------
        # 🏆 Biểu đồ Top 10 Company Name có điểm cảm xúc cao nhất (≥ 100 reviews)
        # -------------------------------
        st.subheader("🏆 Top 10 Companies with Highest Average Sentiment (≥ 100 reviews)")

        top_companies = (
            Reviews.groupby("Company Name")["label_sentiment"]
            .agg(["mean", "count"])
            .reset_index()
            .rename(columns={"mean": "Average Sentiment", "count": "Review Count"})
        )

        # Lọc công ty có ít nhất 100 reviews
        top_companies = top_companies[top_companies["Review Count"] >= 100]

        # Sắp xếp theo điểm cảm xúc trung bình
        top_companies = top_companies.sort_values(by="Average Sentiment", ascending=False).head(10)

        fig = px.bar(
            top_companies,
            x="Average Sentiment",
            y="Company Name",
            orientation='h',
            color="Average Sentiment",
            color_continuous_scale=[[0, "#C0504D"], [0.5, "#F79646"], [1.0, "#4F81BD"]],
            hover_data={"Company Name": True, "Average Sentiment": True, "Review Count": True}
        )
        fig.update_layout(
            xaxis_title="Average Sentiment Score (0 = Negative, 2 = Positive)",
            yaxis_title=None,
            showlegend=False,
            height=500
        )
        st.plotly_chart(fig, use_container_width=True)

    # -------------------------------
    # 📊 Biểu đồ Cluster  
    # -------------------------------
    st.subheader("📊 Cluster Dashboard")

    with st.expander("📂 Click to see Insights", expanded=False):
        # ==== Stopwords và sửa từ sai ====
        custom_stopwords = {
            "are", "if", "each", "with", "some", "stil", "your", "get", "just", "was", "ful", "often",
            "those", "sometimes", "most", "acording", "into", "does", "neds", "quản", "viên", "trị", "fuly",
        }

        wrong_words_dict = {
            "ofice": "office", "bos": "boss", "fel": "feel", "proces": "process",
            "skils": "skills", "employes": "employees", "cofe": "coffee"
        }

        topic_labels = {
            0: "💸 Cluster 1: Compensation & Work Pressure",
            1: "🛠️ Cluster 2: Project & Process",
            2: "👥 Cluster 3: Team Culture & Leadership",
            3: "🏢 Cluster 4: Training & Work-Life Balance",
            4: "🌱 Cluster 5: Growth & Development Opportunities"
        }

        cluster_recommendations = {
            0: "💡 Pay more, reduce overtime, improve management & workspace.",
            1: "💡 Streamline process, empower leaders, support team collaboration.",
            2: "💡 Build positive culture, train kind leaders, retain good teams.",
            3: "💡 Provide more training, promote work-life balance, enhance benefits.",
            4: "💡 Offer growth paths, upskill employees, expand opportunities."
        }

        # ==== Lựa chọn lọc ====
        option = st.radio("🔍 Filter by:", ["Cluster", "Company"], horizontal=True)

        if option == "Cluster":
            cluster_id = st.selectbox("📌 Select cluster:", options=list(topic_labels.keys()), format_func=lambda x: topic_labels[x])
            filtered_df = Reviews[Reviews["cluster"] == cluster_id]
            st.markdown(f"**📌 Recommendation:** {cluster_recommendations[cluster_id]}")

        elif option == "Company":
            company_list = Reviews["Company Name"].dropna().unique().tolist()
            selected_company_name = st.selectbox("🏢 Select company", company_list)
            filtered_df = Reviews[Reviews["Company Name"] == selected_company_name]

            if "cluster" in filtered_df.columns:
                # Lấy cluster phổ biến nhất để gợi ý
                top_cluster = filtered_df["cluster"].value_counts().idxmax()
                cluster_name = topic_labels.get(top_cluster, f"Cluster {top_cluster}")
                st.markdown(f"**📊 Dominant Cluster:** {cluster_name}")
                st.markdown(f"**📌 Recommendation (based on dominant cluster):** {cluster_recommendations.get(top_cluster, 'No suggestion available')}")

        # ==== Làm sạch văn bản ====
        def clean_tokens(text_series):
            tokens = []
            for text in text_series.dropna():
                for word in text.lower().split():
                    word = wrong_words_dict.get(word, word)
                    if word.isalpha() and word not in custom_stopwords:
                        tokens.append(word)
            return tokens

        tokens = clean_tokens(filtered_df["processed_text"])
        word_freq = Counter(tokens).most_common(30)

        # ==== Vẽ Wordcloud ====
        if len(tokens) == 0:
            st.warning("⚠️ No data available for selected filter.")
        else:
            wc = WordCloud(width=800, height=400, background_color="white", colormap='viridis')
            wc.generate_from_frequencies(dict(word_freq))

            st.markdown("### 🌐 Wordcloud (Top 30 words)")
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.imshow(wc, interpolation='bilinear')
            ax.axis("off")
            st.pyplot(fig)

            # ==== Thêm thông tin công ty nếu lọc theo company ====
            if option == "Company":
                try:
                    company_info = Companies[Companies["Company Name"] == selected_company_name].iloc[0]
                    st.markdown("---")
                    st.subheader(f"🏢 Company information: **{company_info['Company Name']}**")
                    st.markdown(f"""
                    - 🔧 **Company type**: {company_info['Company Type']}
                    - 🏭 **Industry**: {company_info['Company industry']}
                    - 👥 **Size**: {company_info['Company size']}
                    
                    - 📅 **Working days**: {company_info['Working days']}
                    - ⏱ **Overtime Policy**: {company_info['Overtime Policy']}
                    - ⭐ **Overall rating**: {company_info['Overall rating']}
                    - 📝 **Number of reviews**: {company_info['Number of reviews']}
                    - 👍 **Recommend to friend**: {company_info['Recommend working here to a friend']}
                    """)

                    company_reviews = Reviews[Reviews["Company Name"] == selected_company_name]
                    sentiment_counts = company_reviews["label_sentiment"].value_counts().to_dict()

                    st.markdown("### 😊 How others review")
                    st.write(f"- 😠 **Negative**: {sentiment_counts.get(0, 0)}")
                    st.write(f"- 😐 **Neutral**: {sentiment_counts.get(1, 0)}")
                    st.write(f"- 😊 **Positive**: {sentiment_counts.get(2, 0)}")

                except Exception as e:
                    st.error(f"⚠️ Error loading company info: {e}")




# ================================
# Tab 3: Sentiment Predictor
# ================================
elif st.session_state.active_tab == "Sentiment":
    st.markdown("""
        <div style="
            background-color: #f5f5f5; 
            padding: 4px; 
            border-radius: 8px; 
            text-align: center; 
            max-width: 800px; 
            margin: auto;
        ">
            <h1 style="color: #333333; margin: 0;">Sentiment Prediction & Clustering Reviews</h1>
        </div>
    """, unsafe_allow_html=True)

    # Load models
    model_Logistic = joblib.load("Logistic_Regression_sentiment_model.pkl")
    model_xgboost = joblib.load("xgboost_sentiment_model.pkl")

    # 🔻 Chọn mô hình
    st.write("### ⚙️ Choose a sentiment prediction model")
    model_choice = st.radio("🔍 Select model:", ("Logistic Regression", "XGBoost"), index=0, horizontal=True)

    # 🔍 Chọn công ty
    st.write("### 🏢 Choose company")
    company_names = [""] + Companies["Company Name"].dropna().unique().tolist()
    selected_company_name = st.selectbox("🔍 Choose company", company_names)
    selected_company_id = None
    if selected_company_name:
        selected_company_id = Companies[Companies["Company Name"] == selected_company_name]["id"].values[0]

    # ✅ Input đánh giá
    st.write("### 👉 Step 1: Choose ratings (1–5)")
    salary = st.slider("💰 Salary & benefits", 1, 5, 3)
    training = st.slider("📚 Training & learning", 1, 5, 3)
    management = st.slider("👨‍💼 Management cares about me", 1, 5, 3)
    culture = st.slider("🎉 Culture & fun", 1, 5, 3)
    workspace = st.slider("🏢 Office & workspace", 1, 5, 3)

    recommend = st.selectbox(
    "🤝 Would you recommend this company to friend?",
    ["---", "Yes", "No"],
    index=0
)


    st.write("### 👉 Step 2: Write your review")
    review_text = st.text_area("✍️ Your opinion", "")

    # 👉 Chuẩn hoá ngôn ngữ về tiếng Anh
    translated_text = fast_translate(review_text)
    process_text= process_text(translated_text)
    # Chuẩn bị input cho mô hình
    input_data = pd.DataFrame([{
        'Salary & benefits': salary,
        'Training & learning': training,
        'Management cares about me': management,
        'Culture & fun': culture,
        'Office & workspace': workspace,
        'pos_count': count_pos_neg_words(translated_text.lower(), positive_words, negative_words)[0],
        'neg_count': count_pos_neg_words(translated_text.lower(), positive_words, negative_words)[1]
    }])


    # Chia nút dự đoán / lưu / xem thông tin
    col_predict, col_cluster, col_save, col_info = st.columns([1, 1, 1, 1])

    with col_predict:
        if st.button("📊 Sentiment ?",help="Predict sentiment based on selected model"):
            if model_choice == "Logistic Regression":
                prediction = model_Logistic.predict(input_data)[0]
            else:
                prediction = model_xgboost.predict(input_data)[0]
            labels = {0: "😠 Negative", 1: "😐 Neutral", 2: "😊 Possitive"}
            st.success(f"✅ Sentiment predict ({model_choice}): **{labels[prediction]}**")

    with col_save:
        # ✅ Khởi tạo session state nếu chưa có
        if "recent_reviews" not in st.session_state:
            st.session_state["recent_reviews"] = []
        if st.button("💾 Save Review", help="Save your review"):
            if not selected_company_name:
                st.error("❌ Please select a company before saving!")
            elif recommend == "---":
                st.error("❌ Please select your recommendation for this company.")
            else:
                # ✅ Calculate average rating
                avg_rating = round((salary + training + management + culture + workspace) / 5, 1)
                now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

                # ✅ Create review record
                review_record = {
                    "🕒 Date time": now,
                    "🏢 Company": selected_company_name,
                    "⭐ Rating": avg_rating,
                    "👍 Recommend?": recommend,
                    "✍️ Review": review_text
                }

                # ✅ Update cache (only keep latest 3)
                st.session_state.recent_reviews.insert(0, review_record)
                st.session_state.recent_reviews = st.session_state.recent_reviews[:3]

                st.success("✅ Your review has been saved!")

    with col_info:
        if st.button("ℹ️ Company Infor.",help="Click to see company information"):
            if not selected_company_name:
                st.error("❌ Please select a company to view its information!")
            else:
                with st.expander(f"🏢 Company Information: {selected_company_name}", expanded=True):
                    company_info = Companies[Companies["Company Name"] == selected_company_name].iloc[0]
                    company_reviews = Reviews[Reviews["Company Name"] == selected_company_name]
                    sentiment_counts = company_reviews["label_sentiment"].value_counts().to_dict()

                    # Xác định emoji theo rating
                    try:
                        rating = float(company_info["Overall rating"])
                        if rating >= 4:
                            emoji = "🌟"
                        elif rating >= 3:
                            emoji = "🙂"
                        elif rating >= 2:
                            emoji = "😐"
                        else:
                            emoji = "😞"
                    except:
                        emoji = "❓"

                    st.markdown(f"""
                    <div style='padding: 10px 20px'>
                        <ul style="list-style-type: none; padding-left: 0; line-height: 1.6;">
                            <li>🔧 <strong>Type:</strong> {company_info['Company Type']}</li>
                            <li>🏭 <strong>Industry:</strong> {company_info['Company industry']}</li>
                            <li>👥 <strong>Size:</strong> {company_info['Company size']}</li>
                            
                            <li>📅 <strong>Working days:</strong> {company_info['Working days']}</li>
                            <li>⏱ <strong>Overtime Policy:</strong> {company_info['Overtime Policy']}</li>
                            <li>⭐ <strong>Overall rating:</strong> {emoji} {company_info['Overall rating']}</li>
                            <li>📝 <strong>Number of reviews:</strong> {company_info['Number of reviews']}</li>
                            <li>👍 <strong>Recommend to friend:</strong> {company_info['Recommend working here to a friend']}</li>
                        </ul>
                        <hr>
                        <h5>😊 <strong>How others review:</strong></h5>
                        <ul style="list-style-type: none; padding-left: 0; line-height: 1.6;">
                            <li>😠 <strong>Negative:</strong> {sentiment_counts.get(0, 0)}</li>
                            <li>😐 <strong>Neutral:</strong> {sentiment_counts.get(1, 0)}</li>
                            <li>😊 <strong>Positive:</strong> {sentiment_counts.get(2, 0)}</li>
                        </ul>
                    </div>
                    """, unsafe_allow_html=True)


    with col_cluster:
        if st.button("🧠 Cluster ?",help="Click to see clustering by LDA model"):
            if not review_text.strip():
                st.error("❌ Please input a review before clustering.")
            else:
                try:
                    # Load vectorizer và LDA model đã huấn luyện
                    vectorizer = joblib.load("vectorizer.pkl")
                    lda_model = joblib.load("lda_model.pkl")

                    # Vector hóa review mới
                    new_vec = vectorizer.transform([review_text])

                    # Dự đoán phân phối topic
                    topic_distribution = lda_model.transform(new_vec)[0]
                    topic_id = topic_distribution.argmax()
                    topic_prob = topic_distribution[topic_id]

                    # Mapping topic ID với nhãn ý nghĩa
                    topic_labels = {
                        0: "💸 Cluster 1:  Compensation & Work Pressure",
                        1: "🛠️ Cluster 2: Project & Process",
                        2: "👥 Cluster 3: Team Culture & Leadership",
                        3: "🏢 Cluster 4: Training & Work-Life Balance",
                        4: "🌱 Cluster 5: Growth & Development Opportunities"
                    }

                    # Gợi ý cải thiện theo cluster
                    cluster_recommendations = {
                        0: "💡 Pay more, reduce overtime, improve management & workspace.",
                        1: "💡 Streamline process, empower leaders, support team collaboration.",
                        2: "💡 Build positive culture, train kind leaders, retain good teams.",
                        3: "💡 Provide more training, promote work-life balance, enhance benefits.",
                        4: "💡 Offer growth paths, upskill employees, expand opportunities."
                    }

                    # Dự đoán phân phối topic
                    topic_distribution = lda_model.transform(new_vec)[0]
                    topic_id = topic_distribution.argmax()
                    topic_prob = topic_distribution[topic_id]

                    # Gán nhãn và gợi ý
                    label = topic_labels.get(topic_id, f"Topic {topic_id}")
                    recommendation = cluster_recommendations.get(topic_id, "No recommendation available.")

                    # Hiển thị kết quả
                    st.success(f"✅ This review belongs to **{label}** with confidence: **{topic_prob:.2f}**")
                    st.info(f"🔍 Recommendation: {recommendation}")

                    # # Hiển thị tất cả phân phối
                    # st.markdown("### 📊 Topic distribution")
                    # for idx, prob in enumerate(topic_distribution):
                    #     this_label = topic_labels.get(idx, f"Topic {idx}")
                    #     st.write(f"- {this_label}: **{prob:.2f}**")

                except Exception as e:
                    st.error(f"❌ Error in clustering: {e}")

# ✅ Hiển thị 10 review mới nhất
if st.session_state["recent_reviews"]:
    st.subheader("🧾 Recent reviews")
    df = pd.DataFrame(st.session_state["recent_reviews"])
    st.dataframe(df)
