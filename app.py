
import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt

# ---------------- PAGE SETTINGS ----------------
st.set_page_config(
    page_title="ReviewSense | Product Insights",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------- CUSTOM MINT-GREEN THEME ----------------
st.markdown("""
<style>
.stApp {
    background: #E9FAF1;
    color: #202442;
}

[data-testid="stHeader"] {
    background: #E9FAF1;
}

[data-testid="stSidebar"] {
    background: #F4FFF8;
    border-right: 1px solid #CDEEDD;
}

.hero {
    background: linear-gradient(110deg, #6845F5, #A54BE8, #F36AA6);
    padding: 30px;
    border-radius: 20px;
    color: white;
    margin-bottom: 24px;
    box-shadow: 0 8px 24px rgba(104, 69, 245, 0.15);
}

.hero h1, .hero p {
    color: white !important;
}

.section-title {
    color: #38237D;
    font-size: 24px;
    font-weight: bold;
    margin: 12px 0;
}

div[data-testid="stMetric"] {
    background: #FFFFFF;
    border: 1px solid #CDEEDD;
    border-radius: 16px;
    padding: 18px;
    box-shadow: 0 4px 14px rgba(44, 150, 100, 0.08);
}

div[data-testid="stMetricLabel"] {
    color: #62627D;
}

div[data-testid="stMetricValue"] {
    color: #603FE0;
}

div.stButton > button {
    background: linear-gradient(100deg, #6845F5, #BD4FDF);
    color: white;
    border: none;
    border-radius: 11px;
    padding: 10px 18px;
    font-weight: bold;
}

div.stButton > button:hover {
    color: white;
    background: #5637CF;
}

div[data-testid="stTextArea"] textarea,
div[data-testid="stTextInput"] input {
    background: white;
    color: #202442;
    border-radius: 10px;
    border: 1px solid #BFE5D0;
}

.info-note {
    background: #F4FFF8;
    border-left: 5px solid #38A878;
    border-radius: 10px;
    padding: 14px;
    color: #245A43;
    margin: 12px 0;
}

h2, h3 {
    color: #38237D;
}
</style>
""", unsafe_allow_html=True)


# ---------------- LOAD MODEL AND DATASET ----------------
@st.cache_resource
def load_model():
    return joblib.load("sentiment_model.pkl")


@st.cache_data
def load_data():
    df = pd.read_csv("data/real_reviews.csv")

    required_columns = [
        "product_id", "product", "review", "rating"
    ]

    missing = [
        column for column in required_columns
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            "Missing dataset columns: " + ", ".join(missing)
        )

    df = df.dropna(
        subset=["product_id", "product", "review", "rating"]
    ).copy()

    df["product"] = df["product"].astype(str)
    df["review"] = df["review"].astype(str)
    df["rating"] = pd.to_numeric(
        df["rating"], errors="coerce"
    )

    df = df.dropna(subset=["rating"])

    return df


try:
    model = load_model()
    data = load_data()

except Exception as error:
    st.error(
        "Unable to load the model or dataset. "
        "Check your project folder and file names."
    )
    st.exception(error)
    st.stop()

if data.empty:
    st.error("The review dataset is empty.")
    st.stop()

with st.spinner("Analyzing customer reviews..."):
    data["predicted_sentiment"] = model.predict(
        data["review"]
    )


# ---------------- SIDEBAR ----------------
with st.sidebar:
    st.markdown(
        "<h2 style='color:#6845F5;'>🛍️ ReviewSense</h2>",
        unsafe_allow_html=True
    )

    st.caption("Product Review Sentiment & Comparison")

    st.divider()

    st.markdown("### 🧭 Explore")
    st.write("📊 Dataset Overview")
    st.write("⚖️ Compare Products")
    st.write("🔎 Explore Reviews")
    st.write("✨ Analyze a Review")

    st.divider()

    st.markdown("""
    <div style="
        background:#E2F8EC;
        padding:16px;
        border-radius:14px;
        color:#245A43;
    ">
        <b>💡 Better Reviews. Better Decisions.</b>
        <p>Discover customer opinions with machine learning.</p>
    </div>
    """, unsafe_allow_html=True)


# ---------------- HEADER ----------------
st.markdown("""
<div class="hero">
    <h1>🛍️ ReviewSense</h1>
    <p>
        Understand customer opinions. Compare products.
        Make smarter shopping decisions.
    </p>
</div>
""", unsafe_allow_html=True)


# ---------------- DATASET OVERVIEW ----------------
st.markdown(
    '<div class="section-title">📊 Dataset Overview</div>',
    unsafe_allow_html=True
)

total_reviews = len(data)
total_products = data["product_id"].nunique()

positive_count = (
    data["predicted_sentiment"] == "positive"
).sum()

negative_count = (
    data["predicted_sentiment"] == "negative"
).sum()

positive_percent = positive_count / total_reviews * 100
negative_percent = negative_count / total_reviews * 100

c1, c2, c3, c4 = st.columns(4)

c1.metric("📝 Total Reviews", f"{total_reviews:,}")
c2.metric("🛍️ Products", f"{total_products:,}")
c3.metric("😊 Positive", f"{positive_percent:.1f}%")
c4.metric("💗 Negative", f"{negative_percent:.1f}%")

st.caption(
    "Sentiment percentages are predictions from the trained model."
)

st.divider()


# ---------------- PRODUCT COMPARISON ----------------
st.markdown(
    '<div class="section-title">⚖️ Compare Two Products</div>',
    unsafe_allow_html=True
)

review_counts = data["product"].value_counts()

eligible_products = (
    review_counts[review_counts >= 3]
    .sort_values(ascending=False)
    .index.tolist()
)

if len(eligible_products) < 2:
    st.warning(
        "Not enough products have at least 3 reviews to compare. "
        "A larger dataset is needed."
    )

else:
    col1, col2 = st.columns(2)

    with col1:
        product1 = st.selectbox(
            "Choose Product 1",
            eligible_products,
            key="product_one"
        )

    remaining_products = [
        product for product in eligible_products
        if product != product1
    ]

    # Keep the second selection valid when Product 1 changes.
    if st.session_state.get("product_two") not in remaining_products:
        st.session_state["product_two"] = remaining_products[0]

    with col2:
        product2 = st.selectbox(
            "Choose Product 2",
            remaining_products,
            key="product_two"
        )

    comparison_rows = []

    for product in [product1, product2]:
        product_data = data[data["product"] == product]
        total = len(product_data)

        positive = (
            product_data["predicted_sentiment"] == "positive"
        ).sum()

        negative = (
            product_data["predicted_sentiment"] == "negative"
        ).sum()

        comparison_rows.append({
            "Product": product,
            "Total Reviews": total,
            "Positive": positive / total * 100,
            "Negative": negative / total * 100,
            "Average Rating": product_data["rating"].mean()
        })

    comparison_df = pd.DataFrame(comparison_rows)

    # PRODUCT SUMMARY CARDS
    card1, card2 = st.columns(2)

    for card, (_, row) in zip(
        [card1, card2],
        comparison_df.iterrows()
    ):
        with card:
            st.markdown(f"#### 🛍️ {row['Product']}")

            st.metric(
                "📝 Customer Reviews",
                f"{int(row['Total Reviews']):,}"
            )

            st.metric(
                "⭐ Average Rating",
                f"{row['Average Rating']:.2f}/5"
            )

            pos_col, neg_col = st.columns(2)

            pos_col.metric(
                "😊 Positive",
                f"{row['Positive']:.1f}%"
            )

            neg_col.metric(
                "💗 Negative",
                f"{row['Negative']:.1f}%"
            )

    # AUTOMATIC COMPARISON RESULT
    st.markdown("### 🏆 Comparison Result")

    positive1 = comparison_df.iloc[0]["Positive"]
    positive2 = comparison_df.iloc[1]["Positive"]

    difference = abs(positive1 - positive2)

    if positive1 > positive2:
        st.success(
            f"🏆 **{product1}** has the higher positive sentiment "
            f"by {difference:.1f} percentage points."
        )
    elif positive2 > positive1:
        st.success(
            f"🏆 **{product2}** has the higher positive sentiment "
            f"by {difference:.1f} percentage points."
        )
    else:
        st.info(
            "Both products have the same positive sentiment "
            "percentage in the analyzed reviews."
        )

    if min(comparison_df["Total Reviews"]) < 10:
        st.warning(
            "One or both products have fewer than 10 reviews. "
            "Treat this comparison as preliminary because a small "
            "sample may not represent all customers."
        )

    # COLORFUL COMPARISON CHART
    st.markdown("### 📈 Sentiment Comparison")

    chart_df = comparison_df.set_index("Product")[
        ["Positive", "Negative"]
    ]

    fig, ax = plt.subplots(figsize=(10, 4))
    fig.patch.set_facecolor("#E9FAF1")
    ax.set_facecolor("#FFFFFF")

    chart_df.plot(
        kind="bar",
        ax=ax,
        color=["#2CC58A", "#F45B87"],
        width=0.65
    )

    ax.set_ylabel("Predicted Sentiment (%)")
    ax.set_xlabel("")
    ax.set_ylim(0, 100)
    ax.grid(axis="y", alpha=0.2)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, ncol=2)
    ax.tick_params(axis="x", rotation=10)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    st.markdown("""
    <div class="info-note">
        💡 <b>How to read the results:</b>
        Positive and negative percentages come from the model's
        predictions. Average rating comes from the dataset.
        The comparison identifies higher positive sentiment;
        it does not guarantee that a product is better.
    </div>
    """, unsafe_allow_html=True)

    # ---------------- REVIEW EXPLORER ----------------
    with st.expander("🔎 Explore Reviews for Selected Products"):

        available_review_products = [product1, product2]

        if st.session_state.get("review_viewer") not in available_review_products:
            st.session_state["review_viewer"] = product1

        chosen_product = st.selectbox(
            "Choose a product to view reviews",
            available_review_products,
            key="review_viewer"
        )

        selected_reviews = data[
            data["product"] == chosen_product
        ].copy()

        filter_col, search_col = st.columns([1, 2])

        with filter_col:
            rating_filter = st.selectbox(
                "Filter by rating",
                [
                    "All ratings",
                    "5 stars",
                    "4 stars",
                    "3 stars",
                    "2 stars",
                    "1 star"
                ],
                key="rating_filter"
            )

        with search_col:
            search_text = st.text_input(
                "Search within reviews",
                placeholder="Try quality, battery, good...",
                key="review_search"
            )

        if rating_filter != "All ratings":
            rating_value = int(rating_filter[0])

            selected_reviews = selected_reviews[
                selected_reviews["rating"] == rating_value
            ]

        if search_text.strip():
            selected_reviews = selected_reviews[
                selected_reviews["review"].str.contains(
                    search_text.strip(),
                    case=False,
                    na=False,
                    regex=False
                )
            ]

        st.write(
            f"**Matching reviews:** {len(selected_reviews):,}"
        )

        if selected_reviews.empty:
            st.info("No matching reviews. Try different filters.")

        else:
            for _, review_row in selected_reviews.head(20).iterrows():
                stars = "⭐" * max(
                    0,
                    min(5, int(round(review_row["rating"])))
                )

                st.markdown(
                    f"**{stars} {review_row['rating']:.0f}/5** "
                    f"· Model prediction: "
                    f"**{str(review_row['predicted_sentiment']).title()}**"
                )

                st.write(review_row["review"])
                st.divider()

            if len(selected_reviews) > 20:
                st.caption("Showing the first 20 matching reviews.")


# ---------------- CUSTOM REVIEW ANALYZER ----------------
st.divider()

st.markdown(
    '<div class="section-title">✨ Analyze Your Own Review</div>',
    unsafe_allow_html=True
)

st.write("Enter a customer review to predict its sentiment.")

review_col, result_col = st.columns([1.2, 1])

with review_col:
    new_review = st.text_area(
        "Customer review",
        placeholder=(
            "Example: The product quality is excellent "
            "and it works perfectly."
        ),
        height=130,
        key="custom_review"
    )

    analyze = st.button(
        "🔍 Analyze Sentiment",
        use_container_width=True
    )

with result_col:
    if analyze:
        if new_review.strip():
            prediction = model.predict(
                [new_review.strip()]
            )[0]

            if prediction == "positive":
                st.success("😊 Positive Sentiment")
            else:
                st.error("💗 Negative Sentiment")

            if hasattr(model, "predict_proba"):
                probabilities = model.predict_proba(
                    [new_review.strip()]
                )[0]

                classes = list(model.classes_)
                predicted_index = classes.index(prediction)

                confidence = probabilities[predicted_index] * 100

                st.metric(
                    "Model confidence",
                    f"{confidence:.1f}%"
                )

                st.caption(
                    "Estimated model probability, not a guarantee "
                    "that the prediction is correct."
                )
        else:
            st.warning("Please enter a review first.")

    else:
        st.markdown("""
        <div style="
            background:linear-gradient(135deg,#E2F8EC,#F1EDFF);
            border:1px solid #CDEEDD;
            border-radius:16px;
            padding:22px;
        ">
            <h3 style="color:#287A56;">💡 Your Review Insights</h3>
            <p style="color:#456858;">
                Your prediction and estimated model confidence
                will appear here.
            </p>
        </div>
        """, unsafe_allow_html=True)


# ---------------- FOOTER ----------------
st.divider()

st.markdown("""
<div style="
    text-align:center;
    color:#547564;
    font-size:12px;
    padding:8px;
">
    <b style="color:#6845F5;">ReviewSense</b> |
    Built with Python, Pandas, Scikit-learn, Matplotlib and Streamlit |
    Amazon review dataset
</div>
""", unsafe_allow_html=True)
