import streamlit as st
import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt

st.set_page_config(layout="centered")
st.title("📈 2025년 질산이온 농도 예측 시스템")

uploaded_file = st.file_uploader("CSV 파일을 업로드하세요", type=["csv"])

if uploaded_file is not None:
    try:
        # CSV 불러오기
        df = pd.read_csv(uploaded_file, encoding='cp949')
        st.success("✅ 파일이 성공적으로 업로드되었습니다!")

        # 컬럼명 확인
        st.subheader("📌 업로드된 CSV 컬럼 목록")
        st.write(df.columns.tolist())

        # 필요한 컬럼 확인
        if '측정날짜' not in df.columns or '질산이온' not in df.columns:
            st.error("❌ '측정날짜' 또는 '질산이온' 컬럼이 없습니다.")
        else:
            # 질산이온을 숫자로 변환 (문자열은 NaN으로)
            df['질산이온'] = pd.to_numeric(df['질산이온'], errors='coerce')
            
            # 날짜 형식을 변환 (잘못된 형식은 NaN으로)
            df['측정날짜'] = pd.to_datetime(df['측정날짜'], errors='coerce')
            
            # NaN 행 제거
            df = df.dropna(subset=['측정날짜', '질산이온'])
            
            # 데이터가 비었는지 확인
            if df.empty:
                st.error("❌ 유효한 데이터가 없습니다. '측정날짜'와 '질산이온' 컬럼의 형식을 확인하세요.")
            else:
                # Prophet용 데이터 준비
                prophet_df = df[['측정날짜', '질산이온']].rename(columns={
                    '측정날짜': 'ds',
                    '질산이온': 'y'
                })

                # 모델 학습
                model = Prophet()
                model.fit(prophet_df)

                # 미래 12개월 예측
                future = model.make_future_dataframe(periods=12, freq='MS')
                forecast = model.predict(future)

                # 예측 결과 그래프
                st.subheader("📊 예측 결과 그래프")
                fig1 = model.plot(forecast)
                st.pyplot(fig1)

                # 2025년 예측 데이터 출력
                st.subheader("📄 2025년 예측 데이터")
                forecast_2025 = forecast[forecast['ds'].dt.year == 2025][['ds', 'yhat', 'yhat_lower', 'yhat_upper']]
                forecast_2025.columns = ['날짜', '예측값', '하한값', '상한값']
                st.dataframe(forecast_2025.reset_index(drop=True))

    except Exception as e:
        st.error(f"⚠️ 오류 발생: {e}")
        st.write("💡 CSV 파일을 확인하세요. '질산이온'에 숫자가 아닌 값(예: '교정')이나 '측정날짜'에 잘못된 날짜 형식이 있을 수 있습니다.")
