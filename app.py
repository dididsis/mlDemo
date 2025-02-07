import streamlit as st
import streamlit_authenticator as stauth
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from PIL import Image
import io
import os
import datetime

import fpdf
from fpdf import FPDF

import yaml
from yaml.loader import SafeLoader

from inference import pred_large_image
from inference import eval
from stitch import stitch_images

user_path = "config.yaml"
CSV_FILE="result_info.csv"
ORIGIN_DIR="image/origin"
STITCHED_DIR="image/stitched"
RESULT_DIR="image/result"
OTHER_DIR="result"
os.makedirs(ORIGIN_DIR, exist_ok=True)
os.makedirs(STITCHED_DIR, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)
os.makedirs(OTHER_DIR, exist_ok=True)

def load_config():
    if not os.path.exists(user_path):
        raise FileNotFoundError(f"{user_path} is not found.")
    with open(user_path, "r", encoding="utf-8") as file:
        config = yaml.load(file, Loader=SafeLoader)
    return config

def create_pdf(id, n, d, l):
    pdf = FPDF()
    pdf.add_font("GenShin", "", "GenShinGothic-Medium.ttf", uni=True)
    pdf.add_page()
    pdf.set_font("GenShin",size = 12)

    pdf.cell(200, 10, txt=f"名称：{n}", ln=True, align='C')
    pdf.cell(200, 10, txt=f"日時：{d}", ln=True, align='C')
    pdf.cell(200, 10, txt=f"場所：{l}", ln=True, align='C')
    current_y = pdf.get_y() + 5
    pdf.image("image/stitched/"+str(id)+"_"+str(d)+"/stitched.png", x=10, y=current_y, w = 80)
    pdf.image("image/result/"+str(id)+"_"+str(d)+"/inference_result.png", x=110, y=current_y, w=80)

    current_y=pdf.get_y() + 10
    #pdf.image_stream(hist_data, x=10, y=current_y, w=80)
    #pdf.image_stream(heatmap_data, x = 110, y=current_y, w=80)
    
    pdf_buffer = io.BytesIO()
    pdf.output(pdf_buffer, 'F')

    pdf_buffer.seek(0)

    return pdf_buffer


def save_config(config):
    with open(user_path, "w", encoding="utf-8") as file:
        yaml.dump(config, file, allow_unicode=True, sort_keys=False)

def save(n, d, l):
    now_str = datetime.datetime.now().strftime("%Y-%m-%d-%H:%M:%S")
    new_record = {
        "id": st.session_state["username"],
        "name": n,
        "dates": d,
        "update": now_str,
        "location": l
    }
    if not os.path.exists(CSV_FILE):
        df = pd.DataFrame([new_record])
        df.to_csv(CSV_FILE, index=False, encoding="utf-8")
    else:
        df = pd.read_csv(CSV_FILE, encoding="utf-8")
        new_df = pd.DataFrame([new_record])
        df = pd.concat([df, new_df], ignore_index=True)
        df.to_csv(CSV_FILE, index=False, encoding="utf-8")
    
    safe_id_dates = (st.session_state["username"] + "_" + d.replace(":", "-").replace("/", "-").replace("\\", "-")).strip()
    origin_subdir = os.path.join(ORIGIN_DIR, safe_id_dates)
    stitched_subdir = os.path.join(STITCHED_DIR, safe_id_dates)
    result_subdir = os.path.join(RESULT_DIR, safe_id_dates)
    other_result_subdir = os.path.join(OTHER_DIR, safe_id_dates)

    os.makedirs(origin_subdir, exist_ok=True)
    os.makedirs(stitched_subdir, exist_ok=True)
    os.makedirs(result_subdir, exist_ok=True)
    os.makedirs(other_result_subdir, exist_ok=True)

    for i, img in enumerate(st.session_state["images"]):
        file_name = str(i)+".png"
        save_path=os.path.join(origin_subdir, file_name)
        with open(save_path, "wb") as f:
            img.save(save_path)
    if isinstance(st.session_state["stitched_img"], np.ndarray):
        # OpenCV形式(BGR)→RGB→PIL
        stitched_pil = Image.fromarray(st.session_state["stitched_img"][:, :, ::-1])
        stitched_pil.save(os.path.join(stitched_subdir, "stitched.png"))
    elif isinstance(st.session_state["stitched_img"], Image.Image):
        st.session_state["stitched_img"].save(os.path.join(stitched_subdir, "stitched.png"))

    if isinstance(st.session_state["pred"], np.ndarray):
        # OpenCV形式(BGR)→RGB→PIL
        result_pil = Image.fromarray(st.session_state["pred"].astype(np.uint8), mode='L')
        result_pil.save(os.path.join(result_subdir, "inference_result.png"))
    elif isinstance(st.session_state["pred"], Image.Image):
        st.session_state["pred"].save(os.path.join(result_subdir, "inference_result.png"))
    
    other_result_file = os.path.join(other_result_subdir, "extra_result.txt")
    with open(other_result_file, "w", encoding="utf-8") as f:
        f.write("ここに検査結果の追加情報などを追記する\n")
    
    st.info("保存完了")

def mypage():
    st.subheader("My Page")
    

def history(target_id):
    st.subheader("検査履歴")

    df = pd.read_csv("result_info.csv", encoding="utf-8")

    df_filtered = df[df["id"] == target_id].copy()

    if df_filtered.empty:
        st.warning(f"検査履歴はありません")

    col1,col2,col3,col4 = st.columns([2,2,2,1])
    col1.write("実施日時")
    col2.write("場所")
    col3.write("名称")
    col4.write("詳細")

    for i, row in df_filtered.iterrows():
        c1,c2,c3,c4 = st.columns([2,2,2,1])

        
        #print(row)
        c1.write(row["dates"])
        c2.write(row["location"])
        c3.write(row["name"])

        if c4.button("詳細", key=f"btn_{i}"):
            st.session_state['r']=row
            st.session_state['page']='detail'
            #st.write(row["id"]+row["dates"])

def detail():
    row=st.session_state['r']
    st.write(f"名称：{row['name']}")
    st.write(f"日時：{row['dates']}")
    st.write(f"場所：{row['location']}")
    simage=Image.open("image/stitched/"+row['id']+"_"+row['dates']+"/stitched.png")
    pimage=Image.open("image/result/"+row['id']+"_"+row['dates']+"/inference_result.png")
    pimg=np.array(pimage)
    coll, colr = st.columns(2)
    with coll:
        st.image(simage)
    with colr:
        st.image(pimage)
    data=eval(pimg)
    st.write(f"平均：{data[3]}")
    colh, colhe = st.columns(2)
    with colh:
        fig, ax = plt.subplots()
        ax.hist(data[1], bins=20, color="blue", alpha=0.7)
        #ax.set_title("Histogram Example (Matplotlib)")
        st.pyplot(fig)
    with colhe:
        figg, axg = plt.subplots(figsize=(7,5))
        heatmap = axg.imshow(data[2], cmap="viridis", aspect="auto")
        figg.colorbar(heatmap, ax=axg)
        #axg.set_title()
        st.pyplot(figg)

def main_app():
    st.subheader("検査")
    if "nex" not in st.session_state:
        st.session_state.nex = False
    if "pr" not in st.session_state:
        st.session_state.pr = False
    if "result" not in st.session_state:
        st.session_state.result = False
    col1, col2 = st.columns([1,1], gap='small')

    with col1:
        st.subheader("(1)画像アップロード")
        uploaded_files =st.file_uploader(
            "画像ファイルを選択",
            type=["png", "jpg", "jpeg"],
            accept_multiple_files=True
        )

        if st.button("next"):
            st.session_state.nex = True
        if st.session_state.nex:
            if not uploaded_files:
                st.warning("ファイルがアップロードされていません")
            else:
                st.session_state["images"] = [Image.open(file) for file in uploaded_files]
                #print(images)
                st.session_state["stitched_img"] = stitch_images(st.session_state["images"])
                st.success("完了")
        
    with col2:
        st.subheader("(2)画像プレビュー")
        if "stitched_img" in st.session_state and st.session_state["stitched_img"] is not None:
            st.image(st.session_state["stitched_img"])
        else:
            st.info("画像がありません")

    col3, col4 = st.columns([1,1], gap='small')

    with col3:
        st.subheader("(3)推論")
        if "stitched_img" in st.session_state:
            if st.button("推論を実行"):
                st.session_state.pr=True
            if st.session_state.pr:
                pred_img=pred_large_image(st.session_state["stitched_img"])
                st.image(pred_img)
                st.session_state["pred"]=pred_img
        else:
            st.info("画像がないため、推論できません")

    with col4:
        st.subheader("(4)検査結果")
        if st.button("結果を見る"):
            st.session_state.result=True
        if st.session_state.result:
            data=eval(st.session_state["pred"])
            st.write(f"平均：{data[3]}")
            colh, colhe = st.columns(2)
            with colh:
                fig, ax = plt.subplots()
                ax.hist(data[1], bins=20, color="blue", alpha=0.7)
                #ax.set_title("Histogram Example (Matplotlib)")
                st.pyplot(fig)
            with colhe:
                figg, axg = plt.subplots(figsize=(7,5))
                heatmap = axg.imshow(data[2], cmap="viridis", aspect="auto")
                figg.colorbar(heatmap, ax=axg)
                #axg.set_title()
                st.pyplot(figg)
            name=st.text_input("名称")
            coll, colr = st.columns(2)
            with coll:
                date=st.text_input("日時")
            with colr:
                location=st.text_input("場所")
            st.write(fpdf.__version__)
            if st.button("保存"):
                save(name,date,location)
                pdf_data = create_pdf(st.session_state['username'], name, date, location)
                st.download_button(
                    label="Download",
                    data=pdf_data,
                    file_name="test.pdf",
                    mime="application/pdf"
                )

def chat():
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            if message.get("is_image", False):
                st.image(message["content"])
            else:
                st.markdown(message["content"])

    if not any(msg["role"] == "assistant" for msg in st.session_state.messages):
        st.session_state.messages.append({
            "role": "assistant",
            "content": "画像をアップロードしてください",
            "is_image": False
        })
        img_files =st.file_uploader(
            "画像ファイルを選択",
            type=["png", "jpg", "jpeg"],
            accept_multiple_files=True
        )
        if img_files is not None:
            st.session_state.messages.append({
                "role":"user",
                "content":img_files,
                "is_image":True
            })

    if user_input := st.chat_input("reply"):
        st.session_state.messages.append({"role":"user","content":user_input})
    
    response = f"Echo: {user_input}"

    st.session_state.messages.append({"role": "assistant", "content": response})
    with st.chat_message("user"):
        st.markdown(user_input)
    with st.chat_message("assistant"):
        st.markdown(response)
    

def main():
    st.set_page_config(layout="wide")
    st.title("ML Demo")

    config = load_config()
    authenticator = stauth.Authenticate(
        config['credentials'],
        config['cookie']['name'],
        config['cookie']['key'],
        config['cookie']['expiry_days']
    )

    #name, authentication_status = 
    authenticator.login('main')
    
    if st.session_state['authentication_status'] is False:
        st.error("ユーザー名またはパスワードが正しくありません")

    if st.session_state['authentication_status'] is None:
        st.warning("ユーザー名とパスワードを入力してください")

    if st.session_state['authentication_status']:
        st.sidebar.title(f"こんにちは、{st.session_state['name']}さん")
        if "page" not in st.session_state:
            st.session_state['page']='inspect'
        with st.sidebar:
            #if st.button("My Page"):
             #   st.session_state['page']='mypage'
            if st.button("検査"):
                st.session_state['page']='inspect'
            if st.button("検査履歴"):
                st.session_state['page']='history'
            if st.button("チャット"):
                st.session_state['page']='chat'

        if st.session_state['page'] is 'mypage':
            mypage()
        if st.session_state['page'] is 'inspect':
            main_app()
        if st.session_state['page'] is 'detail':
            detail()
        if st.session_state['page'] is 'chat':
            chat()
        if st.session_state['page'] is 'history':
            #print(st.session_state)
            history(st.session_state['username'])

        authenticator.logout("Log out", "sidebar")
        
    else:
        st.write("---")
        st.write("新規登録がまだの方はこちら:")

        if "show_signup" not in st.session_state:
            st.session_state["show_signup"] = False

        if st.button("新規登録"):
            st.session_state["show_signup"] = not st.session_state["show_signup"]

        if st.session_state["show_signup"]:
            st.subheader("新規ユーザー登録")

            new_user_id = st.text_input("新ユーザーID")
            new_user_name = st.text_input("新ユーザー名")
            new_password = st.text_input("パスワード", type = "password")
            new_email = st.text_input("email")

            if st.button("登録"):
                if new_user_id in config["credentials"]["usernames"]:
                    st.error("このユーザーIDは既に存在します")
                else:
                    credentials = {
                        "usernames": {
                            new_user_id: {
                                "name": new_user_name,
                                "password": new_password,
                                "email": new_email
                            }
                        }
                    }
                    hashed_pw = stauth.Hasher.hash_passwords(credentials=credentials)

                    config["credentials"]["usernames"][new_user_id]=hashed_pw["usernames"][new_user_id]

                    save_config(config)

                    st.success(f"ユーザー登録完了")

    

if __name__ == "__main__":
    main()