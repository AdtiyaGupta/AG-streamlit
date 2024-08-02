import streamlit as st

def main():
    tab_names = ["Tab 1", "Tab 2", "Tab 3", "Tab 4", "Tab 5"]
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
        st.header("Tab 1 Content")
    elif selected_tab == tab_names[1]:
        st.header("Tab 2 Content")
    elif selected_tab == tab_names[2]:
        st.header("Tab 3 Content")
    elif selected_tab == tab_names[3]:
        st.header("Tab 4 Content")
    elif selected_tab == tab_names[4]:
        st.header("Tab 5 Content")

if __name__ == "__main__":
    main()
