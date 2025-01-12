import os
import datetime
import requests
import fitz  
import streamlit as st
from dotenv import load_dotenv
from audio_recorder_streamlit import audio_recorder
from groq import Groq

load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")
client = Groq(api_key=groq_api_key)
#-----------------


#-----------------
@st.cache_data
def process_cv_from_pdf(file_path):
    try:
        doc = fitz.open(file_path)
        text = ""
        for page in doc:
            text += page.get_text()
        return text.strip()
    except Exception as e:
        st.error(f"Error reading PDF file: {e}")
        return None

def classify_cv_data_with_llm(cv_text):
    API_URL = "https://api.groq.com/openai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {groq_api_key.strip()}",
        "Content-Type": "application/json"
    }
    prompt = (
        f"Here is a CV extracted from a PDF file. Please classify the key information "
        f"into skills, experience, and projects:\n\n{cv_text}\n\nClassified Information:"
    )
    payload = {
        "model": "llama3-8b-8192",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 400,
        "temperature": 0.6,
    }
    try:
        response = requests.post(API_URL, headers=headers, json=payload)
        response.raise_for_status()
        return response.json()['choices'][0]['message']['content']
    except requests.exceptions.RequestException as e:
        st.error(f"API Error: {e}")
        return None

def generate_dynamic_question(previous_answer, cv_text, all_questions_and_answers, position, language):
    API_URL = "https://api.groq.com/openai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {groq_api_key.strip()}",
        "Content-Type": "application/json"
    }
    
    if language == "عربي":
        prompt = (
            f"هذه بيانات السيرة الذاتية:\n\n{cv_text}\n\n"
            f"المرشح يتقدم لشغل وظيفة: {position}\n\n"
            f"هذه هي الأسئلة والإجابات السابقة:\n\n{all_questions_and_answers}\n\n"
            f"بناءً على الإجابة الأخيرة، قم بإنشاء سؤال متابعة ذو صلة بالوظيفة والسيرة الذاتية:\n\n"
            f"الإجابة الأخيرة: {previous_answer}\n\n"
            f"أرجع السؤال فقط دون أي نص إضافي."
        )
    else:
        prompt = (
            f"Here is the CV data:\n\n{cv_text}\n\n"
            f"The candidate is applying for the position of: {position}\n\n"
            f"Here are the previous questions and answers:\n\n{all_questions_and_answers}\n\n"
            f"Based on the last answer, generate a follow-up question that is relevant to the position and the CV:\n\n"
            f"Last Answer: {previous_answer}\n\n"
            f"Return only the question without any additional text."
        )
    
    payload = {
        "model": "llama3-8b-8192",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 100,
        "temperature": 0.6,
    }
    
    try:
        response = requests.post(API_URL, headers=headers, json=payload)
        response.raise_for_status()
        question = response.json()['choices'][0]['message']['content'].strip()
        return question
    except requests.exceptions.RequestException as e:
        st.error(f"API Error: {e}")
        return None
    
    
def save_audio_file(audio_bytes, file_extension="wav"):
    if len(audio_bytes) < 1000:  
        st.warning("الملف الصوتي قصير جدًا أو فارغ. يرجى التسجيل مرة أخرى.")
        return None
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    file_name = f"audio_{timestamp}.{file_extension}"
    with open(file_name, "wb") as f:
        f.write(audio_bytes)
    return file_name

def audio_to_text(filepath):
    try:
        with open(filepath, "rb") as file:
            translation = client.audio.translations.create(
                file=(filepath, file.read()),
                model="whisper-large-v3",
            )
        return translation.text
    except Exception as e:
        st.error(f"Error transcribing audio: {e}")
        return None

def generate_feedback_with_score(questions_and_answers, language):
    API_URL = "https://api.groq.com/openai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {groq_api_key}",
        "Content-Type": "application/json"
    }
    
    prompt = (
        "Here is a mock interview transcript. Please analyze it and provide the following:\n"
        "1. A score out of 10 based on the candidate's performance.\n"
        "2. A list of strengths.\n"
        "3. A list of weaknesses.\n"
        "4. Personalized tips to improve interview performance in the future.\n\n"
        f"Transcript:\n\n{questions_and_answers}\n\n"
        "Please format the feedback as follows:\n"
        "**Score:** X/10\n"
        "**Strengths:**\n- Point 1\n- Point 2\n\n"
        "**Weaknesses:**\n- Point 1\n- Point 2\n\n"
        "**Tips for Improvement:**\n- Tip 1\n- Tip 2\n"
    )
    
    if language == "عربي":
        prompt = (
            "هذا نص لمقابلة وهمية. يرجى تحليلها وتقديم التالي:\n"
            "1. تقييم من 10 بناءً على أداء المرشح.\n"
            "2. قائمة بنقاط القوة.\n"
            "3. قائمة بنقاط الضعف.\n"
            "4. نصائح مخصصة لتحسين أداء المقابلة في المستقبل.\n\n"
            f"النص:\n\n{questions_and_answers}\n\n"
            "يرجى تنسيق التقييم كما يلي:\n"
            "**التقييم:** X/10\n"
            "**نقاط القوة:**\n- نقطة 1\n- نقطة 2\n\n"
            "**نقاط الضعف:**\n- نقطة 1\n- نقطة 2\n\n"
            "**نصائح للتحسين:**\n- نصيحة 1\n- نصيحة 2\n"
            "يجب عليك الرد باللغه العربية فقط"
        )
    
    payload = {
        "model": "llama3-8b-8192",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 500,
        "temperature": 0.7,
    }
    try:
        response = requests.post(API_URL, headers=headers, json=payload)
        response.raise_for_status()
        return response.json()['choices'][0]['message']['content']
    except requests.exceptions.RequestException as e:
        st.error(f"API Error: {e}")
        return None

def upload_cv_page():
    st.title("Welcome to the Sure platform!")
    
    uploaded_file = st.file_uploader("# Upload your CV (PDF)", type="pdf")
    if uploaded_file is not None:
        with open("uploaded_cv.pdf", "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        cv_text = process_cv_from_pdf("uploaded_cv.pdf")
        if cv_text:
            classified_data = classify_cv_data_with_llm(cv_text)
            st.subheader("Summary of your CV:")
            st.write(classified_data)

            st.subheader("Select the Position you are applying for:")
            
            positions = [
                "Software Engineer",
                "Data Scientist",
                "Frontend Developer",
                "Backend Developer",
                "DevOps Engineer",
                "Machine Learning Engineer",
                "Product Manager",
                "UI/UX Designer",
                "Other"
            ]
            
            selected_position = st.selectbox("Choose a position:", positions)
            
            if selected_position == "Other":
                selected_position = st.text_input("Please specify the position:")
            
            if selected_position:
                st.session_state.selected_position = selected_position
                
                language = st.selectbox("Choose the interview language:", ["English", "عربي"])
                st.session_state.language = language
                
                if st.button("Start the Interview"):
                    st.session_state.step = "interview"
                    st.session_state.classified_data = classified_data
                    st.rerun()

def interview_page():
    st.title("Your interview has started with Sure")
    
    if "selected_position" in st.session_state:
        st.write(f"**Position Applied For:** {st.session_state.selected_position}")
    else:
        st.error("Position not selected. Please go back and select a position.")
        return 
    
    if "language" in st.session_state:
        st.write(f"**Interview Language:** {st.session_state.language}")
    else:
        st.error("Language not selected. Please go back and select a language.")
        return 
    
    if "questions" not in st.session_state:
        st.session_state.questions = ["Tell me about yourself."] if st.session_state.language == "English" else ["حدثني عن نفسك."]
    if "answers" not in st.session_state:
        st.session_state.answers = []
    if "current_question" not in st.session_state:
        st.session_state.current_question = st.session_state.questions[0]
    if "progress" not in st.session_state:
        st.session_state.progress = 0

    st.write(f"## Current Question: {st.session_state.current_question}")

    st.write("### Record your answer:")
    audio_bytes = audio_recorder()

    user_answer = None
    if audio_bytes:
        st.audio(audio_bytes, format="audio/wav")
        file_path = save_audio_file(audio_bytes)
        if file_path:
            transcription = audio_to_text(file_path)
            user_answer = transcription

    if st.button("Submit Answer"):
        if user_answer:
            st.session_state.answers.append(user_answer)
            st.session_state.questions.append(st.session_state.current_question)
            st.session_state.progress += 1

            previous_answer = st.session_state.answers[-1]
            cv_text = st.session_state.classified_data
            position = st.session_state.selected_position
            language = st.session_state.language
            all_questions_and_answers = "\n".join(
                f"Q: {q}\nA: {a}" for q, a in zip(st.session_state.questions, st.session_state.answers)
            )

            if st.session_state.language == "English" and st.session_state.progress == 5:
                st.write("You have completed 5 questions in English. Would you like to switch to Arabic?")
                if st.button("Switch to Arabic"):
                    st.session_state.language = "عربي"
                    st.session_state.current_question = "حدثني عن نفسك."  # تغيير السؤال الأول بالعربية
                    st.rerun()

            new_question = generate_dynamic_question(previous_answer, cv_text, all_questions_and_answers, position, language)
            if new_question:
                st.session_state.current_question = new_question
        else:
            st.warning("Please provide an answer.")

    if st.button("End Interview"):
        questions_and_answers = "\n".join(
            f"Q: {q}\nA: {a}" for q, a in zip(st.session_state.questions, st.session_state.answers)
        )
        st.write("### Interview Summary")
        st.write(questions_and_answers)

        feedback = generate_feedback_with_score(questions_and_answers, st.session_state.language)
        st.write("### Feedback for Improvement")
        st.write(feedback)

        st.session_state.questions = []
        st.session_state.answers = []
        st.session_state.current_question = "Tell me about yourself." if st.session_state.language == "English" else "حدثني عن نفسك."
        st.session_state.progress = 0
def main2():
    if 'step' not in st.session_state:
        st.session_state.step = "upload"

    if st.session_state.step == "upload":
        upload_cv_page()
    elif st.session_state.step == "interview":
        interview_page()

def main():
    if "page" not in st.session_state:
        st.session_state.page = "home"

    if st.session_state.page == "train4interview":
        main2()
    else:
        home_page()

def home_page():
    st.image("surelogo.jpg", width=1000)  

    st.title("Welcome to SURE | شُور")
    st.markdown("### Revolutionizing Recruitment with AI-Powered Features")
    st.markdown("---")
    st.subheader("")

    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button("Train for an Interview"):
            st.session_state.page = "train4interview"
            st.rerun()

    with col2:
        if st.button("Applicant Portal"):
            st.info("""
            إذا كنت ترغب في استخدام بوابة المتقدمين، يرجى التواصل مع فريق شور عبر الرابط التالي:
            [تواصل معنا](https://bind.link/@suure)
            """)

    with col3:
        if st.button("HR Portal"):
            st.info("""
            إذا كنت ترغب في استخدام بوابة الموارد البشرية، يرجى التواصل مع فريق شور عبر الرابط التالي:
            [تواصل معنا](https://bind.link/@suure)
            """)
if __name__ == "__main__":
    main()
