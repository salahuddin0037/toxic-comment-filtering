import streamlit as st
from toxic_comment_filtering_main import ToxicCommentClassifier
import time

# Set page configuration
st.set_page_config(
    page_title="Toxic Comment Classifier",
    page_icon="⚠️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
def local_css():
    st.markdown("""
    <style>
        .stTextArea textarea {
            height: 200px;
        }
        .toxic-result {
            background-color: #ffebee;
            border-radius: 5px;
            padding: 15px;
            margin: 10px 0;
        }
        .non-toxic-result {
            background-color: #e8f5e9;
            border-radius: 5px;
            padding: 15px;
            margin: 10px 0;
        }
        .progress-bar {
            height: 20px;
            background-color: #e0e0e0;
            border-radius: 10px;
            margin: 5px 0;
        }
        .progress-fill {
            height: 100%;
            border-radius: 10px;
            background-color: #ff5252;
            text-align: center;
            color: white;
            line-height: 20px;
            font-size: 12px;
        }
        .header {
            color: #d32f2f;
        }
    </style>
    """, unsafe_allow_html=True)

def main():
    local_css()
    
    # Initialize classifier
    classifier = ToxicCommentClassifier()
    
    st.title("🚨 Toxic Comment Classifier")
    st.markdown("""
    This tool analyzes text and detects different types of toxicity including:
    - Toxic
    - Severe Toxic
    - Obscene
    - Threat
    - Insult
    - Identity Hate
    """)
    
    # Example comments
    example_comments = {
        "Clean comment": "I appreciate your perspective on this matter.",
        "Mild toxicity": "You're not very smart, are you?",
        "Severe toxicity": "I hope you die, you worthless piece of trash!",
        "Obscene language": "Go f*** yourself you stupid b****!",
        "Gaming toxicity": "kys noob, you're trash at this game l2p"
    }
    
    # Sidebar with examples
    with st.sidebar:
        st.header("Try these examples:")
        selected_example = st.selectbox(
            "Select an example comment",
            list(example_comments.keys())
        )
        
        if st.button("Load Example"):
            st.session_state.example_comment = example_comments[selected_example]
    
    # Main input area
    comment = st.text_area(
        "Enter a comment to analyze:",
        value=st.session_state.get("example_comment", ""),
        placeholder="Type or paste your comment here..."
    )
    
    if st.button("Analyze Comment", type="primary"):
        if not comment.strip():
            st.warning("Please enter a comment to analyze")
        else:
            with st.spinner("Analyzing comment..."):
                # Simulate processing time for better UX
                time.sleep(0.5)
                
                # Get predictions
                results = classifier.predict(comment)
                
                # Display overall toxicity
                max_toxicity = max(results.values())
                if max_toxicity > 0.5:
                    st.error("⚠️ Toxic content detected!")
                else:
                    st.success("✅ No toxic content detected")
                
                # Show detailed results
                st.subheader("Detailed Analysis")
                
                for label, score in results.items():
                    # Create a progress bar for each toxicity type
                    st.markdown(f"**{label.replace('_', ' ').title()}**")
                    st.markdown(
                        f"""
                        <div class="progress-bar">
                            <div class="progress-fill" style="width: {score*100}%">
                                {score*100:.1f}%
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
                
                # Show processed text
                with st.expander("View processed text"):
                    st.text(classifier.preprocess_text(comment))
                
                # Show interpretation
                st.subheader("Interpretation")
                if max_toxicity > 0.8:
                    st.warning("This content appears highly toxic and potentially harmful.")
                elif max_toxicity > 0.5:
                    st.warning("This content shows signs of toxicity.")
                else:
                    st.info("This content appears clean with low toxicity signals.")

if __name__ == "__main__":
    main()