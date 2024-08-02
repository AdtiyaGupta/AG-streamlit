import streamlit as st

def main():
    st.set_page_config(
        page_title="Octavian Login",
        page_icon="🚀",
        layout="centered",
        initial_sidebar_state="collapsed"
    )
    
    st.title("Emulet Login")

    col1, col2, col3 = st.columns([0.1, 0.8, 0.1])

    with col2:
        #st.image("astronaut.png", width=200)

        username = st.text_input("Username")
        password = st.text_input("Password", type="password")

        sign_in = st.button("Sign In")

        if sign_in:
            # Add your sign-in logic here
            st.success("Logged in successfully!")

    col4, col5, col6 = st.columns([0.1, 0.8, 0.1])

    with col5:
        st.caption("Powered by preprod/corp")
        st.image("static_logo.png", width=50)

if __name__ == "__main__":
    main()
