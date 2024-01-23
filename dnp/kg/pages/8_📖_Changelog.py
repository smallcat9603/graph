import streamlit as st

st.markdown("""
            ## 240123
            - Optimized "Submit" logic
            - Optimized neo4j server connection parameters
            - Added sandbox perf to chart
            ## 240122
            - Imported data CSV from URL instead of local file
            - Disabled 'Run' for large datasets in sandbox
            ## 240121
            - Fixed save button bug
            ## 240119
            - Replaced st.bar_chart with pyplot
            - Fixed pr bug
            ## 240118
            - "Reset" is always enabled in case of program interruption
            ## 240117
            - Optimized "Reset" procedure
            - Confirmed to reset gdbs status when switching between databases
            - Added author info
            - Added emoji
            ## 240116
            - Added chart and changelog pages
""")
