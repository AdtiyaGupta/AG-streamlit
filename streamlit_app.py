import streamlit as st

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
