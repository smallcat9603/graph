import streamlit as st
import pandas as pd

st.title("Graph Construction Time")
st.write("Offline vs Semi-Online vs Online (DNP, nphrase=50)")
df = pd.DataFrame(
    data=[298.8, 26747.1, 120191.6],
    columns=["Graph Construction Time (ms)"],
    index=["Offline", "Semi-Online", "Online"],
)
tab1, tab2 = st.tabs(["Chart", "Table"])
with tab1:
    st.bar_chart(data=df,
                y="Graph Construction Time (ms)",
    )
with tab2:
    st.dataframe(df)
