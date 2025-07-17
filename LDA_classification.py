import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from gensim import corpora, models
from gensim.models.coherencemodel import CoherenceModel
import pyLDAvis.gensim_models as gensimvis
import pyLDAvis
from wordcloud import WordCloud
import io
import base64
from openai import OpenAI
from tqdm import tqdm
import re
import warnings
warnings.filterwarnings('ignore')

# matplotlib 한글 폰트 설정 (전역)
import platform
import os

def setup_korean_font():
    """matplotlib 한글 폰트 설정"""
    system = platform.system()
    
    if system == "Windows":
        font_paths = [
            "C:/Windows/Fonts/malgun.ttf",
            "C:/Windows/Fonts/NanumGothic.ttf",
            "C:/Windows/Fonts/gulim.ttc",
        ]
    elif system == "Darwin":
        font_paths = [
            "/System/Library/Fonts/AppleSDGothicNeo.ttc",
            "/Library/Fonts/NanumGothic.ttf",
        ]
    else:
        font_paths = [
            "/usr/share/fonts/truetype/nanum/NanumGothic.ttf",
            "/usr/share/fonts/TTF/NanumGothic.ttf",
        ]
    
    for font_path in font_paths:
        if os.path.exists(font_path):
            font_prop = fm.FontProperties(fname=font_path)
            plt.rcParams['font.family'] = font_prop.get_name()
            plt.rcParams['axes.unicode_minus'] = False
            return font_path
    
    return None

# 한글 폰트 설정 실행
korean_font_path = setup_korean_font()

# 언어별 토크나이저 import
try:
    from konlpy.tag import Okt
    korean_available = True
except ImportError:
    korean_available = False
    
try:
    import nltk
    from nltk.corpus import stopwords
    from nltk.tokenize import word_tokenize
    from nltk.stem import WordNetLemmatizer
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)
    english_available = True
except ImportError:
    english_available = False

# 페이지 설정
st.set_page_config(page_title="LDA 토픽 모델링 도구", page_icon="📊", layout="wide")

# 제목
st.title("📊 LDA 토픽 모델링 도구")
st.markdown("---")

# 사이드바 설정
st.sidebar.title("설정")

# 기본 불용어 정의
KOREAN_STOPWORDS = ['및', '이', '그', '등', '수', '의', '에', '를', '을', '가', '으로', '에서', 
                   '하는', '하여', '한다', '솔루션', '목적', '포함', '위해', '단계', '방법', '로부터',
                   '통해', '위한', '따라', '대한', '있는', '있다', '없는', '없다', '같은', '다른',
                   '또한', '또는', '그리고', '하지만', '그러나', '따라서', '그래서']

ENGLISH_STOPWORDS = ['and', 'or', 'but', 'the', 'a', 'an', 'is', 'are', 'was', 'were', 'been', 'be',
                    'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should',
                    'may', 'might', 'must', 'can', 'this', 'that', 'these', 'those', 'i', 'you', 'he',
                    'she', 'it', 'we', 'they', 'me', 'him', 'her', 'us', 'them', 'my', 'your', 'his',
                    'her', 'its', 'our', 'their', 'mine', 'yours', 'hers', 'ours', 'theirs', 'solution',
                    'purpose', 'method', 'step', 'include', 'for', 'from', 'through', 'with', 'by']

# 언어별 토크나이저 함수
def tokenize_korean(text, additional_stopwords=[]):
    if not korean_available:
        st.error("한국어 처리를 위해 konlpy를 설치해주세요.")
        return []
    
    okt = Okt()
    stopwords = KOREAN_STOPWORDS + additional_stopwords
    tokens = okt.nouns(text)
    return [word for word in tokens if word not in stopwords and len(word) > 1]

def tokenize_english(text, additional_stopwords=[]):
    if not english_available:
        st.error("영어 처리를 위해 nltk를 설치해주세요.")
        return []
    
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english')) | set(ENGLISH_STOPWORDS) | set(additional_stopwords)
    
    # 텍스트 전처리
    text = re.sub(r'[^a-zA-Z\s]', '', text.lower())
    tokens = word_tokenize(text)
    
    # 불용어 제거 및 표제어 추출
    tokens = [lemmatizer.lemmatize(word) for word in tokens 
              if word not in stop_words and len(word) > 2]
    
    return tokens

# 세션 상태 초기화
if 'df' not in st.session_state:
    st.session_state.df = None
if 'processed_texts' not in st.session_state:
    st.session_state.processed_texts = None
if 'dictionary' not in st.session_state:
    st.session_state.dictionary = None
if 'corpus' not in st.session_state:
    st.session_state.corpus = None
if 'lda_model' not in st.session_state:
    st.session_state.lda_model = None
if 'coherence_values' not in st.session_state:
    st.session_state.coherence_values = None
if 'model_list' not in st.session_state:
    st.session_state.model_list = None
if 'selected_column' not in st.session_state:
    st.session_state.selected_column = None
if 'language' not in st.session_state:
    st.session_state.language = None
if 'additional_stopwords_list' not in st.session_state:
    st.session_state.additional_stopwords_list = []
if 'topic_names' not in st.session_state:
    st.session_state.topic_names = None
if 'doc_topics' not in st.session_state:
    st.session_state.doc_topics = None

# 1. 파일 업로드
st.header("1. 파일 업로드")
uploaded_file = st.file_uploader("CSV 또는 Excel 파일을 업로드하세요", type=['csv', 'xlsx', 'xls'])

if uploaded_file is not None:
    try:
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file, encoding='utf-8')
        else:
            df = pd.read_excel(uploaded_file)
        
        st.session_state.df = df
        st.success(f"파일이 성공적으로 업로드되었습니다! ({len(df)}행, {len(df.columns)}열)")
        
        # 데이터 미리보기
        st.subheader("데이터 미리보기")
        st.dataframe(df.head())
        
    except Exception as e:
        st.error(f"파일 읽기 오류: {str(e)}")

# 2. 컬럼 선택
if st.session_state.df is not None:
    st.header("2. 분석할 컬럼 선택")
    
    st.warning("⚠️ 주의사항: 선택한 컬럼 내의 모든 데이터는 동일한 언어로 작성되어야 합니다.")
    
    text_columns = st.session_state.df.select_dtypes(include=['object']).columns.tolist()
    
    # 기본값 설정
    default_index = 0
    if st.session_state.selected_column and st.session_state.selected_column in text_columns:
        default_index = text_columns.index(st.session_state.selected_column)
    
    selected_column = st.selectbox("LDA 분석을 수행할 텍스트 컬럼을 선택하세요:", 
                                  text_columns, 
                                  index=default_index,
                                  key="column_selector")
    
    # 세션에 저장
    st.session_state.selected_column = selected_column
    
    if selected_column:
        # 결측치 확인 (원본 데이터를 변경하지 않고 표시만)
        null_count = st.session_state.df[selected_column].isnull().sum()
        if null_count > 0:
            st.warning(f"선택한 컬럼에 {null_count}개의 결측치가 있습니다. 전처리 시 자동으로 제거됩니다.")
        
        # 텍스트 데이터 미리보기
        st.subheader("선택한 컬럼의 텍스트 예시")
        preview_df = st.session_state.df.dropna(subset=[selected_column])
        st.text_area("텍스트 미리보기", value="\n".join(preview_df[selected_column].head(3).tolist()), height=150, label_visibility="collapsed")

# 3. 언어 선택 및 전처리
if st.session_state.df is not None and st.session_state.selected_column:
    st.header("3. 언어 선택 및 전처리")
    
    available_languages = []
    if korean_available and english_available:
        available_languages = ["한국어", "영어"]
    elif korean_available:
        available_languages = ["한국어"]
    elif english_available:
        available_languages = ["영어"]
    
    # 기본값 설정
    default_lang_index = 0
    if st.session_state.language and st.session_state.language in available_languages:
        default_lang_index = available_languages.index(st.session_state.language)
    
    language = st.selectbox("텍스트 언어를 선택하세요:", 
                           available_languages,
                           index=default_lang_index,
                           key="language_selector")
    
    # 세션에 저장
    st.session_state.language = language
    
    if language:
        # 추가 불용어 입력
        st.subheader("추가 불용어 설정")
        
        # 기본 불용어 표시
        if language == "한국어":
            st.text("기본 불용어: " + ", ".join(KOREAN_STOPWORDS[:10]) + "...")
        else:
            st.text("기본 불용어: " + ", ".join(ENGLISH_STOPWORDS[:10]) + "...")
        
        # 기존 추가 불용어가 있다면 표시
        existing_stopwords = ", ".join(st.session_state.additional_stopwords_list) if st.session_state.additional_stopwords_list else ""
        
        additional_stopwords = st.text_input("추가 불용어 (쉼표로 구분):", 
                                           value=existing_stopwords,
                                           key="additional_stopwords")
        additional_stopwords_list = [word.strip() for word in additional_stopwords.split(",") if word.strip()]
        
        # 세션에 저장
        st.session_state.additional_stopwords_list = additional_stopwords_list
        
        # 전처리 실행
        if st.button("텍스트 전처리 실행", key="preprocess_btn"):
            # 결측치 제거된 데이터 사용
            clean_df = st.session_state.df.dropna(subset=[st.session_state.selected_column])
            texts = clean_df[st.session_state.selected_column].astype(str).tolist()
            
            with st.spinner("텍스트 전처리 중..."):
                if language == "한국어":
                    processed_texts = [tokenize_korean(text, additional_stopwords_list) for text in texts]
                else:
                    processed_texts = [tokenize_english(text, additional_stopwords_list) for text in texts]
                
                # 빈 문서 제거
                processed_texts = [text for text in processed_texts if text]
                
                if len(processed_texts) == 0:
                    st.error("전처리 후 유효한 텍스트가 없습니다. 불용어 설정을 확인해주세요.")
                else:
                    st.session_state.processed_texts = processed_texts
                    
                    # 딕셔너리 및 코퍼스 생성
                    dictionary = corpora.Dictionary(processed_texts)
                    corpus = [dictionary.doc2bow(text) for text in processed_texts]
                    
                    st.session_state.dictionary = dictionary
                    st.session_state.corpus = corpus
                    
                    st.success(f"전처리 완료! {len(processed_texts)}개 문서, 단어 수: {len(dictionary)}")
                    
                    # 전처리 결과 미리보기
                    st.subheader("전처리 결과 미리보기")
                    for i, tokens in enumerate(processed_texts[:3]):
                        st.text(f"문서 {i+1}: " + " ".join(tokens[:20]) + "...")
        
        # 전처리 상태 표시
        if st.session_state.processed_texts is not None:
            st.info(f"✅ 전처리 완료됨: {len(st.session_state.processed_texts)}개 문서, {len(st.session_state.dictionary) if st.session_state.dictionary else 0}개 단어")

# 4. 최적 토픽 수 탐색
if st.session_state.processed_texts is not None:
    st.header("4. 최적 토픽 수 탐색")
    
    col1, col2 = st.columns(2)
    with col1:
        start_topics = st.number_input("시작 토픽 수", min_value=2, max_value=20, value=2, key="start_topics")
    with col2:
        end_topics = st.number_input("종료 토픽 수", min_value=3, max_value=30, value=10, key="end_topics")
    
    if st.button("토픽 수 탐색 실행", key="coherence_search_btn"):
        if start_topics >= end_topics:
            st.error("시작 토픽 수는 종료 토픽 수보다 작아야 합니다.")
        else:
            with st.spinner("최적 토픽 수를 탐색 중입니다..."):
                coherence_values = []
                model_list = []
                
                progress_bar = st.progress(0)
                for i, num_topics in enumerate(range(start_topics, end_topics + 1)):
                    model = models.LdaModel(
                        corpus=st.session_state.corpus,
                        id2word=st.session_state.dictionary,
                        num_topics=num_topics,
                        random_state=42,
                        passes=10,
                        alpha='auto',
                        eta='auto'
                    )
                    model_list.append(model)
                    
                    coherence_model = CoherenceModel(
                        model=model,
                        texts=st.session_state.processed_texts,
                        dictionary=st.session_state.dictionary,
                        coherence='c_v'
                    )
                    coherence_values.append(coherence_model.get_coherence())
                    
                    progress_bar.progress((i + 1) / (end_topics - start_topics + 1))
                
                st.session_state.coherence_values = coherence_values
                st.session_state.model_list = model_list
                st.session_state.coherence_start = start_topics
                st.session_state.coherence_end = end_topics
                
                # 그래프 그리기
                fig, ax = plt.subplots(figsize=(10, 6))
                x = range(start_topics, end_topics + 1)
                ax.plot(x, coherence_values, 'b-', marker='o')
                ax.set_xlabel('토픽 수')
                ax.set_ylabel('Coherence Score')
                ax.set_title('최적 토픽 수 탐색')
                ax.grid(True)
                
                # 최고 점수 표시
                max_idx = coherence_values.index(max(coherence_values))
                optimal_topics = start_topics + max_idx
                ax.annotate(f'최적: {optimal_topics}', 
                           xy=(optimal_topics, coherence_values[max_idx]),
                           xytext=(optimal_topics + 1, coherence_values[max_idx] + 0.01),
                           arrowprops=dict(arrowstyle='->', color='red'))
                
                st.pyplot(fig)
                
                st.success(f"탐색 완료! 권장 토픽 수: {optimal_topics} (Coherence: {coherence_values[max_idx]:.4f})")
    
    # 기존 결과가 있다면 표시
    elif st.session_state.coherence_values is not None:
        st.info("✅ 토픽 수 탐색이 완료되었습니다. 아래에서 최종 토픽 수를 선택하세요.")
        
        # 기존 그래프 다시 표시
        fig, ax = plt.subplots(figsize=(10, 6))
        x = range(st.session_state.coherence_start, st.session_state.coherence_end + 1)
        ax.plot(x, st.session_state.coherence_values, 'b-', marker='o')
        ax.set_xlabel('토픽 수')
        ax.set_ylabel('Coherence Score')
        ax.set_title('최적 토픽 수 탐색')
        ax.grid(True)
        
        # 최고 점수 표시
        max_idx = st.session_state.coherence_values.index(max(st.session_state.coherence_values))
        optimal_topics = st.session_state.coherence_start + max_idx
        ax.annotate(f'최적: {optimal_topics}', 
                   xy=(optimal_topics, st.session_state.coherence_values[max_idx]),
                   xytext=(optimal_topics + 1, st.session_state.coherence_values[max_idx] + 0.01),
                   arrowprops=dict(arrowstyle='->', color='red'))
        
        st.pyplot(fig)

# 5. 토픽 수 선택 및 LDA 모델 생성
if st.session_state.coherence_values is not None or st.session_state.processed_texts is not None:
    st.header("5. 최종 토픽 수 선택")
    
    optimal_num = st.number_input("사용할 토픽 수를 입력하세요:", min_value=2, max_value=50, value=3, key="final_topic_num")
    
    if st.button("LDA 모델 생성", key="lda_create_btn"):
        with st.spinner("LDA 모델을 생성 중입니다..."):
            lda_model = models.LdaModel(
                corpus=st.session_state.corpus,
                id2word=st.session_state.dictionary,
                num_topics=optimal_num,
                random_state=42,
                passes=20,
                alpha='auto',
                eta='auto'
            )
            
            st.session_state.lda_model = lda_model
            st.success("LDA 모델이 성공적으로 생성되었습니다!")
    
    # 기존 모델 상태 표시
    if st.session_state.lda_model is not None:
        st.info(f"✅ LDA 모델 생성 완료: {st.session_state.lda_model.num_topics}개 토픽")

# 6. pyLDAvis 시각화
if st.session_state.lda_model is not None:
    st.header("6. 토픽 시각화 (pyLDAvis)")
    
    st.info("💡 원들이 서로 겹치지 않을 때 토픽이 적절하게 분리된 것입니다. 원이 많이 겹친다면 토픽 수를 조정해보세요.")
    
    if st.button("토픽 시각화 생성"):
        with st.spinner("시각화를 생성 중입니다..."):
            try:
                vis = gensimvis.prepare(st.session_state.lda_model, st.session_state.corpus, st.session_state.dictionary)
                html_string = pyLDAvis.prepared_data_to_html(vis)
                st.components.v1.html(html_string, height=800)
            except Exception as e:
                st.error(f"시각화 생성 오류: {str(e)}")

# 7. 토픽별 상위 단어 조회
if st.session_state.lda_model is not None:
    st.header("7. 토픽별 상위 단어")
    
    num_words = st.number_input("토픽별 상위 단어 개수", min_value=10, max_value=500, value=100, key="num_words_display")
    st.info("💡 토픽명을 정하기 위해서는 100~150개의 단어를 확인하는 것이 적절합니다.")
    
    # 자동으로 토픽별 단어 표시 (버튼 없이)
    st.subheader("토픽별 주요 단어")
    for i, topic in st.session_state.lda_model.show_topics(num_topics=-1, num_words=num_words, formatted=False):
        with st.expander(f"토픽 {i} - 상위 {num_words}개 단어"):
            words = [word for word, prob in topic]
            st.write(", ".join(words))

# 8. GPT API를 통한 토픽명 생성
if st.session_state.lda_model is not None:
    st.header("8. AI 기반 토픽명 생성")
    
    openai_api_key = st.text_input("OpenAI API 키를 입력하세요:", type="password", key="openai_key")
    domain_context = st.text_area("전체 데이터의 도메인 설명 (선택사항):", 
                                  placeholder="예: 이 데이터는 AI 관련 특허 문서들입니다.",
                                  key="domain_context")
    
    # GPT에 전달할 단어 개수 설정
    gpt_num_words = st.number_input("GPT 토픽명 생성에 사용할 단어 개수", 
                                    min_value=20, max_value=200, value=50, 
                                    key="gpt_num_words",
                                    help="토픽명 생성을 위해 GPT에게 전달할 각 토픽당 단어 개수")
    
    if st.button("토픽명 생성", key="topic_naming_btn") and openai_api_key:
        try:
            # OpenAI 클라이언트 초기화 (새 버전 방식)
            client = OpenAI(api_key=openai_api_key)
            
            # 각 토픽의 상위 단어 추출 (사용자 설정 개수 사용)
            topics_words = []
            for i, topic in st.session_state.lda_model.show_topics(num_topics=-1, num_words=gpt_num_words, formatted=False):
                words = [word for word, prob in topic]
                topics_words.append(f"토픽 {i}: {', '.join(words)}")
            
            # GPT 프롬프트 생성
            prompt = f"""
다음은 LDA 토픽 모델링 결과입니다. 각 토픽에 대해 적절한 토픽명을 한국어로 제안해주세요.

{domain_context}

토픽별 주요 단어:
{chr(10).join(topics_words)}

요청사항:
1. 각 토픽에 대해 간결하고 의미있는 토픽명을 제안
2. 토픽 간 내용 중복 여부 분석
3. 토픽 개수 조정 필요성에 대한 의견 제시

형식:
토픽 0: [토픽명]
토픽 1: [토픽명]
...

분석 의견:
[토픽 중복성 및 개수 조정 필요성에 대한 의견]
"""
            
            with st.spinner("AI가 토픽명을 생성 중입니다..."):
                # 새 OpenAI API 방식
                response = client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=1000,
                    temperature=0.3
                )
                
                ai_response = response.choices[0].message.content
                st.success("AI 토픽명 생성 완료!")
                st.write(ai_response)
                
                # 세션에 저장
                st.session_state.topic_names = ai_response
                
        except Exception as e:
            st.error(f"AI 토픽명 생성 오류: {str(e)}")
            st.info("💡 해결방법: OpenAI API 키가 유효한지 확인하거나, 구버전 사용을 원한다면 `pip install openai==0.28.1`을 실행하세요.")
    
    # 기존 토픽명이 있다면 표시
    if st.session_state.topic_names is not None:
        st.subheader("생성된 토픽명")
        st.write(st.session_state.topic_names)

# 9. 각 토픽별 대표 문서
if st.session_state.lda_model is not None:
    st.header("9. 토픽별 대표 문서")
    
    num_docs = st.number_input("토픽별 대표 문서 개수", min_value=1, max_value=5, value=3, key="num_docs_display")
    
    # 문서 토픽 할당이 없다면 계산
    if st.session_state.doc_topics is None:
        # 각 문서의 주요 토픽 결정
        doc_topics = []
        for doc_bow in st.session_state.corpus:
            topic_probs = st.session_state.lda_model.get_document_topics(doc_bow)
            if topic_probs:
                top_topic = max(topic_probs, key=lambda x: x[1])[0]
                doc_topics.append(top_topic)
            else:
                doc_topics.append(-1)
        st.session_state.doc_topics = doc_topics
    
    # 자동으로 대표 문서 표시
    st.subheader("토픽별 대표 문서")
    
    # 결측치가 제거된 원본 데이터 사용
    clean_df = st.session_state.df.dropna(subset=[st.session_state.selected_column]).reset_index(drop=True)
    clean_df['LDA_토픽'] = st.session_state.doc_topics
    
    for topic_num in range(st.session_state.lda_model.num_topics):
        with st.expander(f"토픽 {topic_num} 대표 문서 ({num_docs}개)"):
            topic_docs = clean_df[clean_df['LDA_토픽'] == topic_num]
            
            if len(topic_docs) > 0:
                sample_docs = topic_docs.head(num_docs)
                for idx, (_, row) in enumerate(sample_docs.iterrows()):
                    st.write(f"**문서 {idx+1}:**")
                    st.write(row[st.session_state.selected_column])
                    st.write("---")
            else:
                st.write("이 토픽에 할당된 문서가 없습니다.")

# 10. 워드클라우드 생성
if st.session_state.lda_model is not None:
    st.header("10. 토픽별 워드클라우드")
    
    if st.button("워드클라우드 생성", key="wordcloud_btn"):
        # 한글 폰트 경로 설정
        import platform
        import os
        from matplotlib import font_manager
        
        def get_korean_font_path():
            system = platform.system()
            
            # Windows
            if system == "Windows":
                font_paths = [
                    "C:/Windows/Fonts/malgun.ttf",  # 맑은 고딕
                    "C:/Windows/Fonts/NanumGothic.ttf",  # 나눔고딕
                    "C:/Windows/Fonts/gulim.ttc",  # 굴림
                ]
            # macOS
            elif system == "Darwin":
                font_paths = [
                    "/System/Library/Fonts/AppleSDGothicNeo.ttc",  # 애플 SD 고딕 Neo
                    "/Library/Fonts/NanumGothic.ttf",  # 나눔고딕
                    "/System/Library/Fonts/Helvetica.ttc",
                ]
            # Linux
            else:
                font_paths = [
                    "/usr/share/fonts/truetype/nanum/NanumGothic.ttf",  # 나눔고딕
                    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
                    "/usr/share/fonts/TTF/NanumGothic.ttf",
                ]
            
            # 존재하는 첫 번째 폰트 반환
            for font_path in font_paths:
                if os.path.exists(font_path):
                    return font_path
            
            return None
        
        korean_font = get_korean_font_path()
        
        if korean_font is None:
            st.warning("⚠️ 한글 폰트를 찾을 수 없습니다. 영문으로만 표시됩니다.")
            st.info("""
            💡 한글 폰트 설치 방법:
            - **Windows**: 기본적으로 맑은고딕이 있어야 합니다
            - **macOS**: 시스템 폰트 사용
            - **Linux**: `sudo apt-get install fonts-nanum` 실행
            """)
        
        # matplotlib 한글 폰트 설정
        if korean_font:
            font_prop = font_manager.FontProperties(fname=korean_font)
            plt.rcParams['font.family'] = font_prop.get_name()
            plt.rcParams['axes.unicode_minus'] = False  # 마이너스 기호 깨짐 방지
        
        for i, topic in st.session_state.lda_model.show_topics(num_words=100, formatted=False):
            word_freq = {word: prob for word, prob in topic}
            
            # 워드클라우드 생성
            wordcloud = WordCloud(
                font_path=korean_font,  # 한글 폰트 지정
                background_color='white',
                width=800,
                height=400,
                max_words=100,
                colormap='viridis',  # 색상 테마
                relative_scaling=0.5,
                random_state=42
            ).generate_from_frequencies(word_freq)
            
            # 플롯 생성
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.imshow(wordcloud, interpolation='bilinear')
            ax.axis('off')
            
            # 한글 제목 설정
            if korean_font:
                ax.set_title(f'토픽 {i} 워드클라우드', fontproperties=font_prop, fontsize=16, pad=20)
            else:
                ax.set_title(f'Topic {i} WordCloud', fontsize=16, pad=20)
            
            st.pyplot(fig)
            plt.close()  # 메모리 해제
            
        # matplotlib 폰트 설정 복원
        plt.rcParams.update(plt.rcParamsDefault)

# 11. 결과 다운로드
if st.session_state.lda_model is not None:
    st.header("11. 결과 다운로드")
    
    if st.button("결과 파일 생성", key="download_btn"):
        # 문서 토픽 할당이 없다면 계산
        if st.session_state.doc_topics is None:
            doc_topics = []
            for doc_bow in st.session_state.corpus:
                topic_probs = st.session_state.lda_model.get_document_topics(doc_bow)
                if topic_probs:
                    top_topic = max(topic_probs, key=lambda x: x[1])[0]
                    doc_topics.append(top_topic)
                else:
                    doc_topics.append(-1)
            st.session_state.doc_topics = doc_topics
        
        # 결과 데이터프레임 생성 (결측치 제거된 데이터 사용)
        result_df = st.session_state.df.dropna(subset=[st.session_state.selected_column]).reset_index(drop=True)
        result_df['LDA_토픽'] = st.session_state.doc_topics
        
        # GPT 토픽명이 있다면 추가
        if st.session_state.topic_names is not None:
            # GPT 응답에서 토픽명 추출 (간단한 파싱)
            topic_name_map = {}
            lines = st.session_state.topic_names.split('\n')
            for line in lines:
                if '토픽' in line and ':' in line:
                    try:
                        topic_num = int(line.split('토픽')[1].split(':')[0].strip())
                        topic_name = line.split(':')[1].strip()
                        topic_name_map[topic_num] = topic_name
                    except:
                        continue
            
            result_df['토픽명'] = result_df['LDA_토픽'].map(topic_name_map)
        
        # CSV 파일로 변환
        csv_buffer = io.StringIO()
        result_df.to_csv(csv_buffer, index=False, encoding='utf-8-sig')
        csv_data = csv_buffer.getvalue()
        
        st.download_button(
            label="📥 결과 CSV 다운로드",
            data=csv_data,
            file_name="LDA_분석결과.csv",
            mime="text/csv",
            key="download_csv_btn"
        )
        
        st.success("결과 파일이 생성되었습니다!")
        
        # 결과 미리보기
        st.subheader("결과 데이터 미리보기")
        st.dataframe(result_df.head())

# 사이드바 정보
st.sidebar.markdown("---")
st.sidebar.subheader("📖 사용 가이드")
st.sidebar.markdown("""
1. CSV/Excel 파일 업로드
2. 분석할 텍스트 컬럼 선택
3. 언어 선택 및 불용어 설정
4. 최적 토픽 수 탐색
5. LDA 모델 생성
6. 결과 분석 및 다운로드
""")

st.sidebar.markdown("---")
st.sidebar.subheader("⚙️ 요구사항")
st.sidebar.markdown("""
**Python 패키지:**
- streamlit
- pandas
- gensim
- pyLDAvis
- wordcloud
- openai>=1.0.0
- konlpy (한국어)
- nltk (영어)
- matplotlib

**OpenAI API 참고:**
- 새 버전: openai>=1.0.0
- 구 버전: openai==0.28.1
""")