import streamlit as st

from streamlit_option_menu import option_menu
import home, trending, test, your, about
from pathlib import Path


st.set_page_config(
        page_title="MWANACHUO",
        page_icon='✔'
)

current_dir = Path(__file__).parent if "__file__" in locals() else Path.cwd()
css_file = current_dir / "styles" / "main.css"

with open(css_file) as f:
    st.markdown("<style>{}</style>".format(f.read()), unsafe_allow_html=True)


class MultiApp:
    def __init__(self):
        self.apps = []

    def add_app(self, title, func):
        self.apps.append({
            "title": title,
            "function": function,
        })

    def run():
        # app = st.sidebar(
        with st.sidebar:        
            app = option_menu(
                menu_title='MWANACHUO ',
                options=['Home','Login','Post','Profile','More'],
                icons=['house-door-fill','box-arrow-in-right','wechat','person-fill','three-dots'],
                menu_icon='app-indicator',
                default_index=1,
                styles={
                    "container": {"padding": "5!important","background-color":'white'},
        "icon": {"color": "black", "font-size": "23px"}, 
        "nav-link": {"color":"black","font-size": "20px", "text-align": "left", "margin":"0px", "--hover-color": "grey"},
        "nav-link-selected": {"background-color": "#02ab21"},}
                
                )

        
        if app == "Home":
            trending.app()
        if app == "Login":
            test.app()    
        if app == "Post":
            home.app()        
        if app == 'Profile':
            your.app()
        if app == 'More':
            about.app()    
             
          
             
    run()            

def create_tabs(tab_names):
    selected_tab = st.selectbox("Select Tab", tab_names)
    st.session_state['selected_tab'] = selected_tab

def display_tab_content(selected_tab):
    if selected_tab == "Introduction":
        st.header("Introduction")
        st.write("This is the introduction tab. Provide relevant information here.")
    elif selected_tab == "Data Ingestion":
        st.header("Data Ingestion")
        st.write("This is the data ingestion tab. Describe the process here.")
    elif selected_tab == "Data Transform":
        st.header("Data Transform")
        st.write("This is the data transformation tab. Explain the steps involved.")
    elif selected_tab == "Auto Train ML Model":
        st.header("Auto Train ML Model")
        st.write("This is the auto train ML model tab. Describe the process here.")
    elif selected_tab == "Freeze the Learning":
        st.header("Freeze the Learning")
        st.write("This is the freeze the learning tab. Explain the process here.")

def main():
    tab_names = ["Introduction", "Data Ingestion", "Data Transform", "Auto Train ML Model", "Freeze the Learning"]
    selected_tab = st.session_state.get('selected_tab', tab_names[0])

    create_tabs(tab_names)
    display_tab_content(selected_tab)

if __name__ == "__main__":
    main()
