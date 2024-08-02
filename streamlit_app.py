import streamlit as st

# Define tab names
tab_names = ["Tab 1", "Tab 2", "Tab 3", "Tab 4", "Tab 5"]

# Create tabs
tabs = st.tabs(tab_names)

# Content for each tab
with tabs[0]:
    st.header("Tab 1 Content")
    # Add your content here

with tabs[1]:
    st.header("Tab 2 Content")
    # Add your content here

with tabs[2]:
    st.header("Tab 3 Content")
    # Add your content here

with tabs[3]:
    st.header("Tab 4 Content")
    # Add your content here

with tabs[4]:
    st.header("Tab 5 Content")
    # Add your content here

# Custom CSS to style tabs (optional)
st.markdown("""
<style>
.stTabs {
  display: flex;
  justify-content: space-around;
}
.stTab button {
  background-color: #f2f2f2;
  padding: 10px 20px;
  border: none;
  cursor: pointer;
  transition: background-color 0.3s;
}
.stTab button:hover {
  background-color: #ddd;
}
</style>
""", unsafe_allow_html=True)
