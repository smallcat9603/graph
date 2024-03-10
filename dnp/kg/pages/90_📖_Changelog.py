import streamlit as st

st.markdown("""
            ## 230310
            - Added airport data
            - Constructed graph using edgelist with edge attributes
            ## 230225
            - Added my embedding t-SNE plot
            ## 230224
            - Added blogcatalog data
            ## 230223
            - Added similarity option PMI
            ## 230222
            - Fix bugs
            ## 230221
            - Added page of RW-based Embedding
            ## 230216
            - Added newfood (Kaggle) dataset
            - Dropped all constraints when "Reset"
            - Used node/relationship properties to project graph
            ## 230209
            - Added ML model
            - Used optuna to tune hyperparameters
            ## 230208
            - Merged datasets to DATA
            ## 230207
            - Modulized
            - Showed FastRP and node2vec embeddings and added t-SNE plots
            ## 230206
            - Added pages of Project and Node Similarity
            ## 230205
            - Used t-SNE to plot high-dimension embedding
            ## 230204
            - Added get_nodes_relationships_csv(file)
            - Used cypher file to construct graph
            ## 240203
            - Dropped constraint if exists (otherwise indexes exist even when nodes have been already deleted)
            - Deleted pages.lib.param
            ## 240202
            - Added Embedding
            ## 240131
            - Added nDCG chart for different model sizes
            ## 240130
            - Optimized parameter form (added On-the-fly and hid Verbose)
            - Added On-the-fly (spaCy)
            ## 240129
            - Modulized
            ## 240127
            - Converted r.common to stringlist for offline (A cypher bug of using split() for STRING relationship properties?)
            ## 240126
            - Converted n.phrase to stringlist (still bug for r.common) for offline 
            ## 240124
            - Added nDCG chart
            ## 240123
            - Optimized "Submit" logic
            - Optimized neo4j server connection parameters
            - Added sandbox perf to chart
            - Added result figs for different nphrase
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
