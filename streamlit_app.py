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


    background_image_url = st.image("https://wpamelia.com/wp-content/uploads/2019/02/background-black-colors-952670.jpg")

     # Create a container with a full-viewport class
    container = st.container()
    
    # Add CSS style to the container
    container.markdown(
        
        <style>
        '''
        .full-viewport {
            position: fixed;
            z-index: -1;
            width: 100%;
            height: 100%;
            top: 0;
            left: 0;
            background-image: url('""" + background_image_url + """');
            background-size: cover;
            background-repeat: no-repeat;
        }
        </style>'''
        ,
        unsafe_allow_html=False,  # Safe way to inject CSS )
    
    # Add the full-viewport class to the container
    container.add_class("full-viewport")

    col4, col5, col6 = st.columns([0.1, 0.8, 0.1])

    with col5:
        st.caption("Powered by preprod/corp")
        st.image("static_logo.png", width=300)

    
  

if __name__ == "__main__":
    main()
