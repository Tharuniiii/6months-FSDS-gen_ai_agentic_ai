import streamlit as st 

st.title("My First Stresmlit app")
st.write("welcome! this is app claculates the square of a number.")
st.header("Please Select a number")#creating a interactive slider
number = st.slider("pick a number",0,100,25)# min max and default numbers
#calculate and display result
st.subheader("result")
squared_number = number * number
st.write(f"The square of **{number}** is **{squared_number}**,")


