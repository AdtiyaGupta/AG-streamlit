import streamlit as st

def main():
    tab_names = ["Introduction", "Data Ingestion", "Data Transformation", "Auto Train ML Model", "Freeze the Learning"]
    selected_tab = st.session_state.get('selected_tab', tab_names[0])

    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        if col1.button(tab_names[0]):
            st.session_state['selected_tab'] = tab_names[0]
    with col2:
        if col2.button(tab_names[1]):
            st.session_state['selected_tab'] = tab_names[1]
    with col3:
        if col3.button(tab_names[2]):
            st.session_state['selected_tab'] = tab_names[2]
    with col4:
        if col4.button(tab_names[3]):
            st.session_state['selected_tab'] = tab_names[3]
    with col5:
        if col5.button(tab_names[4]):
            st.session_state['selected_tab'] = tab_names[4]

    st.write(f"Selected tab: {selected_tab}")

    # Content based on selected tab
    if selected_tab == tab_names[0]:
        st.header("Introduction")
    elif selected_tab == tab_names[1]:
        st.header("Data Ingestion")
    elif selected_tab == tab_names[2]:
        st.header("Data Transform")
    elif selected_tab == tab_names[3]:
        st.header("Auto Train Ml Model")
    elif selected_tab == tab_names[4]:
        st.header("Freeze the Learning")

if __name__ == "__main__":
    main()
