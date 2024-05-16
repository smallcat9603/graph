import streamlit as st
import numpy
from pages.lib import flow

@st.cache_data
def run(query):
    return st.session_state["gds"].run_cypher(query)

def get_node_labels():
    query = """
    CALL db.labels()
    """
    result = run(query)
    return list(result["label"])

def get_node_properties(node_label):
    query = f"""
    MATCH (n:{node_label})
    RETURN DISTINCT keys(n) AS properties
    """
    result = run(query)
    result = result["properties"][0]
    result = [value for value in result if isinstance(value, str) and not value.startswith("__csv_")]

    result_numeric = []
    for prop in result:
        value = run(f"MATCH (n:{node_label}) RETURN n.`{prop}` AS propertyValue")["propertyValue"][0]
        if isinstance(value, (float, int, numpy.int64)) or (isinstance(value, list) and isinstance(value[0], (float, int, numpy.int64))):
            result_numeric.append(prop)

    return result_numeric

def get_relationship_types():
    query = """
    CALL db.relationshipTypes()
    """
    result = run(query)
    return list(result["relationshipType"])

def get_relationship_properties(rel):
    query = f"""
    MATCH ()-[r:{rel}]->()
    RETURN DISTINCT keys(r) AS properties
    """
    result = run(query)
    result = result["properties"][0]
    result = [value for value in result if isinstance(value, str) and not value.startswith("__csv_")]

    result_numeric = []
    for prop in result:
        value = run(f"MATCH ()-[r:{rel}]->() RETURN r.`{prop}` AS propertyValue")["propertyValue"][0]
        if isinstance(value, (float, int, numpy.int64)) or (isinstance(value, list) and isinstance(value[0], (float, int, numpy.int64))):
            result_numeric.append(prop)

    return result_numeric

def create_constraint(constraint_name):
    query = f"""
    CREATE CONSTRAINT {constraint_name} IF NOT EXISTS 
    For (a:Article) REQUIRE a.url IS UNIQUE;
    """
    run(query)

# convert string to value
def post_process():
    query = f"""
    MATCH (n) WHERE n.pr0 IS NOT NULL
    SET n.pr0 = toFloat(n.pr0)
    """
    run(query)
    query = f"""
    MATCH (n) WHERE n.pr1 IS NOT NULL
    SET n.pr1 = toFloat(n.pr1)
    SET n.pr2 = toFloat(n.pr2)
    SET n.pr3 = toFloat(n.pr3)
    """
    run(query)
    query = f"""
    MATCH (n) WHERE n.phrase IS NOT NULL
    SET n.salience = apoc.convert.fromJsonList(n.salience)
    """
    run(query)
    query = f"""
    MATCH ()-[r:CONTAINS]-() WHERE r.rank IS NOT NULL
    SET r.rank = toInteger(r.rank)
    SET r.weight = toInteger(r.weight)
    SET r.score = toFloat(r.score)
    """
    run(query)
    query = f"""
    MATCH ()-[r]-() WHERE type(r) =~ 'SIMILAR_.*'
    SET r.score = toFloat(r.score)
    """
    run(query)
    query = f"""
    MATCH (n) WHERE n.phrase IS NOT NULL
    SET n.phrase = replace(n.phrase, "[", "")
    SET n.phrase = replace(n.phrase, "]", "")
    SET n.phrase = split(n.phrase, ",")
    """
    run(query)    
    query = f"""
    MATCH ()-[r:CORRELATES]-() WHERE r.common IS NOT NULL
    SET r.common = replace(r.common, "[", "")
    SET r.common = replace(r.common, "]", "")
    WITH r, r.common AS common
    SET r.common = split(common, ",")
    """
    run(query) 

def import_graph_data(DATA):
    nodes, relationships = flow.get_nodes_relationships_csv(f"{st.session_state['dir']}{DATA}.csv")
    dir = st.session_state["dir"]
    query = "CALL apoc.import.csv(["
    for idx, node in enumerate(nodes):
        query += f"{{fileName: '{dir}{DATA}.nodes.{node}.csv', labels: ['{node}']}}, "
        if idx == len(nodes)-1:
            query = query[:-2] + "], ["
    for idx, relationship in enumerate(relationships):
        query += f"{{fileName: '{dir}{DATA}.relationships.{relationship}.csv', type: '{relationship}'}}, "
        if idx == len(relationships)-1:
            query = query[:-2] + "], {})"
    result = run(query)
    post_process()
    return result

def load_data_url(url):
    query = f"""
    CALL apoc.periodic.iterate(
        "LOAD CSV WITH HEADERS FROM '{url}' AS row
        RETURN row",
        "MERGE (a:Article {{name: row.id, url: row.url}})
        SET a.grp = CASE WHEN 'occupation' IN keys(row) THEN row.occupation ELSE null END
        SET a.grp1 = CASE WHEN 'nationality' IN keys(row) THEN row.nationality ELSE null END
        WITH a
        CALL apoc.load.html(a.url, {{
        title: 'title',
        h2: 'h2',
        body: 'body p'
        }})
        YIELD value
        WITH a,
            reduce(texts = '', n IN range(0, size(value.body)-1) | texts + ' ' + coalesce(value.body[n].text, '')) AS body,
            value.title[0].text AS title
        SET a.body = body, a.title = title",
        {{batchSize: 5, parallel: true}}
    )
    YIELD batches, total, timeTaken, committedOperations
    RETURN batches, total, timeTaken, committedOperations
    """
    run(query)

def set_phrase_salience_properties_csv(file, query_node=False):
    if query_node == False:
        query = f"""
        LOAD CSV WITH HEADERS FROM "{file}" AS row
        WITH row
        WHERE row._labels = ":Article"
        MATCH (a:Article {{name: row.name}}) WHERE a.processed IS NULL
        SET a.processed = true
        SET a.phrase = apoc.convert.fromJsonList(row.phrase)
        SET a.salience = apoc.convert.fromJsonList(row.salience)
        RETURN COUNT(a) AS Processed
        """
    else:
        query = f"""
        LOAD CSV WITH HEADERS FROM "{file}" AS row
        WITH row
        WHERE row._labels = ":Query"
        MATCH (q:Query {{name: row.name}})
        SET q.processed = true
        SET q.phrase = apoc.convert.fromJsonList(row.phrase)
        SET q.salience = apoc.convert.fromJsonList(row.salience)
        """
    return run(query)

def set_phrase_salience_properties_gcp(gcp_api_key, query_node=False):
    if query_node == False:
        query = f"""
        CALL apoc.periodic.iterate(
        "MATCH (a:Article)
        WHERE a.processed IS NULL
        RETURN a",
        "CALL apoc.nlp.gcp.entities.stream([item in $_batch | item.a], {{
            nodeProperty: 'body',
            key: '{gcp_api_key}'
        }})
        YIELD node, value
        SET node.processed = true
        WITH node, value
        UNWIND value.entities AS entity
        SET node.phrase = coalesce(node.phrase, []) + entity['name']
        SET node.salience = coalesce(node.salience, []) + entity['salience']",
        {{batchMode: "BATCH_SINGLE", batchSize: 10}})
        YIELD batches, total, timeTaken, committedOperations
        RETURN batches, total, timeTaken, committedOperations
        """
    else:
        query = f"""
        MATCH (q:Query)
        CALL apoc.nlp.gcp.entities.stream(q, {{
        nodeProperty: 'body',
        key: '{gcp_api_key}'
        }})
        YIELD node, value
        SET node.processed = true
        WITH node, value
        UNWIND value.entities AS entity
        SET node.phrase = coalesce(node.phrase, []) + entity['name']
        SET node.salience = coalesce(node.salience, []) + entity['salience']
        """
    return run(query)

def set_phrase_salience_properties_spacy(language, word_class, pipeline_size, n, query_node=False):      
    nlp = flow.get_nlp(language, pipeline_size)
    if query_node == False:
        query = """
        MATCH (a:Article) 
        RETURN a.url as url, a.body AS text
        """
        result = run(query)

        narticles = len(result["url"])
        for i in range(narticles):
            url = result["url"][i]
            text = result["text"][i]
            top_keywords = flow.extract_keywords(nlp, text, word_class, n)
            phrase = [item[0] for item in top_keywords]
            salience = [item[1] for item in top_keywords]
            query = f"""
            MATCH (a:Article {{url: "{url}"}})
            SET a.processed = true
            SET a.phrase = {phrase}
            SET a.salience = {salience}
            """
            run(query)
    else:
        query = """
        MATCH (q:Query) 
        RETURN q.url as url, q.body AS text
        """
        result = run(query)

        narticles = len(result["url"])
        for i in range(narticles):
            url = result["url"][i]
            text = result["text"][i]
            top_keywords = flow.extract_keywords(nlp, text, word_class, n)
            phrase = [item[0] for item in top_keywords]
            salience = [item[1] for item in top_keywords]
            query = f"""
            MATCH (q:Query {{url: "{url}"}})
            SET q.processed = true
            SET q.phrase = {phrase}
            SET q.salience = {salience}
            """
            run(query)
            
    return result

def create_noun_article_relationships(nphrase, query_node=False):
    if query_node == False:
        query = f"""
        MATCH (a:Article)
        WHERE a.processed IS NOT NULL
        FOREACH (word IN a.phrase[0..{nphrase}] |
        MERGE (n:Noun {{name: word}})
        MERGE (a)-[r:CONTAINS]-(n)
        SET r.rank = apoc.coll.indexOf(a.phrase, word) + 1
        SET r.score = a.salience[apoc.coll.indexOf(a.phrase, word)]
        SET r.weight = {nphrase} - apoc.coll.indexOf(a.phrase, word)
        )
        """
    else:
        query = f"""
        MATCH (q:Query)
        WHERE q.processed IS NOT NULL
        FOREACH (word IN q.phrase[0..{nphrase}] |
        MERGE (n:Noun {{name: word}})
        MERGE (q)-[r:CONTAINS]-(n)
        SET r.rank = apoc.coll.indexOf(q.phrase, word) + 1
        SET r.score = q.salience[apoc.coll.indexOf(q.phrase, word)]
        SET r.weight = {nphrase} - apoc.coll.indexOf(q.phrase, word)
        )
        """
    run(query)

def create_query_node(name, url):
    query = f"""
    MERGE (q:Query {{name: "{name}", url: "{url}"}})
    WITH q
    CALL apoc.load.html(q.url, {{
    title: "title",
    h2: "h2",
    body: "body p"
    }})
    YIELD value
    WITH q,
        reduce(texts = "", n IN range(0, size(value.body)-1) | texts + " " + coalesce(value.body[n].text, "")) AS body,
        value.title[0].text AS title
    SET q.body = body, q.title = title
    RETURN q.title, q.body
    """
    run(query)

def create_article_article_relationships(nphrase, query_node=False):
    if query_node == False:
        query = f"""
        MATCH (a1:Article), (a2:Article)
        WHERE a1 <> a2 AND any(x IN a1.phrase[0..{nphrase}] WHERE x IN a2.phrase[0..{nphrase}])
        MERGE (a1)-[r:CORRELATES]-(a2)
        SET r.common = [x IN a1.phrase[0..{nphrase}] WHERE x IN a2.phrase[0..{nphrase}]]
        """
    else:
        query = f"""
        MATCH (q:Query), (a:Article)
        WHERE any(x IN q.phrase[0..{nphrase}] WHERE x IN a.phrase[0..{nphrase}])
        MERGE (q)-[r:CORRELATES]-(a)
        SET r.common = [x IN q.phrase[0..{nphrase}] WHERE x IN a.phrase[0..{nphrase}]]
        """
    run(query)

def get_graph_statistics():
    query = f"""
    CALL apoc.meta.stats()
    YIELD nodeCount, relCount, labels, relTypesCount
    RETURN nodeCount, relCount, labels, relTypesCount
    """
    return run(query)

def save_graph_data(data):
    # no bulkImport: all in one
    # use bulkImport to generate multiple files categorized by node label and relationship type
    query = f"""
    CALL apoc.export.csv.all("{data}.csv", {{}}) 
    """
    result_allinone = run(query)
    query = f"""
    CALL apoc.export.csv.all("{data}.csv", {{bulkImport: true}}) 
    """
    result_bulkimport = run(query)

    st.write(result_allinone)
    # st.write(result_bulkimport)

def interact_knn(emb, QUERY_DICT):
    col1, col2 = st.columns(2)
    with col1:
        query_node = st.selectbox("Query node", tuple(QUERY_DICT.keys()), key=f"{emb}_col1")
    with col2:
        limit = st.selectbox("Limit", ("5", "10", "15", "20"), key=f"{emb}_col2")
    st.write("The top-" + limit + " similar nodes for query " + query_node + " are ranked as follows (" + emb + ")")
    if emb == "emb_frp":
        m = "FASTRP"
    elif emb == "emb_n2v":
        m = "NODE2VEC"
    elif emb == "emb_hgn":
        m = "HASHGNN"
    query = f"""
    MATCH (q:Query)-[r:SIMILAR_{m}]-(a:Article) WHERE q.name = "{query_node}"
    RETURN q.name AS Query, a.name AS Article, a.url AS URL, a.grp AS Group, a.grp1 AS Group1, r.score AS Similarity
    ORDER BY Similarity DESC
    LIMIT {limit}
    """   
    result = run(query)
    tab01, tab02 = st.tabs(["Chart", "Table"]) 
    with tab01:
        flow.plot_similarity(result, query_node, emb, limit)
    with tab02:
        st.write(result)
    return query

def interact_node_similarity(QUERY_DICT):
    col1, col2, col3 = st.columns(3)
    with col1:
        query_node = st.selectbox("Query node", tuple(QUERY_DICT.keys()))
    with col2:
        similarity_method = st.selectbox("Similarity method", ("JACCARD", "OVERLAP", "COSINE", "PPR"))
    with col3:
        limit = st.selectbox("Limit", ("5", "10", "15", "20"))
    st.write("The top-" + limit + " similar nodes for query " + query_node + " are ranked as follows (" + similarity_method + ")")
    if similarity_method == "PPR":
        query = f"""
        MATCH (q:Query)-[r:CORRELATES]-(a:Article) WHERE q.name = "{query_node}"
        RETURN q.name AS Query, a.name AS Article, a.url AS URL, a.pr{str(int(query_node.split("-")[-1])-1)} AS Similarity
        ORDER BY Similarity DESC
        LIMIT {limit}
        """ 
    else:
        query = f"""
        MATCH (q:Query)-[r:SIMILAR_{similarity_method}]-(a:Article) WHERE q.name = "{query_node}"
        RETURN q.name AS Query, a.name AS Article, a.url AS URL, r.score AS Similarity
        ORDER BY Similarity DESC
        LIMIT {limit}
        """   
    result = run(query)
    tab01, tab02 = st.tabs(["Chart", "Table"])
    with tab01:
        flow.plot_similarity(result, query_node, similarity_method, limit)
    with tab02:
        st.write(result)
    return query

def interact_multiple_queries(QUERY_DICT):
    col1, col2, col3 = st.columns(3)
    with col1:
        queries = list(QUERY_DICT.keys())
        query_nodes = st.multiselect("Query node", queries, queries[:2])
    with col2:
        similarity_method = st.selectbox("Similarity method", ("JACCARD", "OVERLAP", "COSINE", "PPR"), key="sm")
    with col3:
        limit = st.selectbox("Limit", ("5", "10", "15", "20"), key="lim")
    st.write("The top-" + limit + " similar nodes for queries " + ', '.join(query_nodes) + " are ranked as follows (" + similarity_method + ")")
    ppr_attr = ["pr" + str(int(item.replace("C-", ""))-1) for item in query_nodes]
    if similarity_method == "PPR":
        query = f"""
        MATCH (q:Query)-[r:CORRELATES]-(a:Article) WHERE q.name IN {query_nodes}
        RETURN COLLECT(q.name) AS Query, a.name AS Article, REDUCE(s = 0, pr IN {ppr_attr} | s + a[pr]) AS Similarity
        ORDER BY Similarity DESC
        LIMIT {limit}
        """ 
    else:
        query = f"""
        MATCH (q:Query)-[r:SIMILAR_{similarity_method}]-(a:Article) WHERE q.name IN {query_nodes}
        RETURN COLLECT(q.name) AS Query, a.name AS Article, SUM(r.score) AS Similarity
        ORDER BY Similarity DESC
        LIMIT {limit}
        """   
    result = run(query)
    tab01, tab02 = st.tabs(["Chart", "Table"])
    with tab01:
        flow.plot_similarity(result, ', '.join(query_nodes), similarity_method, limit)
    with tab02:
        st.write(result)
    return query

def interact_related_articles():
    noun = st.text_input("Keyword", "環境")
    query = f"""
    MATCH (n:Noun)-[]-(a:Article) WHERE n.name CONTAINS "{noun}"
    WITH DISTINCT a AS distinctArticle, n
    RETURN n.name AS Keyword, COUNT(distinctArticle) AS articleCount, COLLECT(distinctArticle.name) AS articles
    ORDER BY articleCount DESC
    """   
    result = run(query)
    st.write(result)
    return query

def interact_common_keywords():
    query = f"""
    MATCH (n:Noun)-[]-(a:Article)
    RETURN n.name AS Keyword, COUNT(a) AS articleCount, COLLECT(a.name) AS articles
    ORDER BY articleCount DESC
    """   
    result = run(query)
    st.write(result)
    return query

def interact_naive_by_rank():
    query = """
    MATCH (q:Query)-[r:CONTAINS]-(n:Noun)-[c:CONTAINS]-(a:Article)
    RETURN q.name AS Query, a.name AS Article, a.url AS URL, a.grp AS Group, a.grp1 AS Group1, collect(n.name) AS Common, SUM((1.0/r.rank)*(1.0/c.rank)) AS Similarity 
    ORDER BY Query, Similarity DESC
    LIMIT 10
    """
    result = run(query)
    st.write(result)
    return query

def interact_naive_by_salience():
    query = """
    MATCH (q:Query)-[r:CORRELATES]-(a:Article)
    WITH q, r, a, reduce(s = 0.0, word IN r.common | 
    s + q.salience[apoc.coll.indexOf(q.phrase, word)] + a.salience[apoc.coll.indexOf(a.phrase, word)]) AS Similarity
    RETURN q.name AS Query, a.name AS Article, a.url AS URL, a.grp AS Group, a.grp1 AS Group1, r.common, Similarity 
    ORDER BY Query, Similarity DESC
    LIMIT 10
    """
    result = run(query)
    st.write(result)
    return query

def free_up_db():
    query = """
    MATCH (n) DETACH DELETE n
    """
    run(query)

    query = f"""
    SHOW CONSTRAINTS
    YIELD name
    RETURN name
    """
    result = run(query)["name"]
    for constraint in result:
        query = f"""
        DROP CONSTRAINT {constraint} IF EXISTS
        """
        run(query)


def runFile(file_cypher):
    query = f"""
    CALL apoc.cypher.runFile("{file_cypher}")
    """
    run(query)

def update_emb_result():
    query = """
    MATCH (n)
    RETURN n.name AS name, n.emb_frp AS emb_frp, n.emb_n2v AS emb_n2v
    """
    result = st.session_state["gds"].run_cypher(query)
    return result

def drop_emb():
    query = """
    MATCH (n) REMOVE n.emb_frp
    """
    st.session_state["gds"].run_cypher(query)  
    query = """
    MATCH (n) REMOVE n.emb_n2v
    """
    st.session_state["gds"].run_cypher(query) 

def update_emb_status(emb):
    query = f"""
    MATCH (n) WHERE n.{emb} IS NOT NULL
    RETURN n
    """
    result = st.session_state["gds"].run_cypher(query) 
    status = True 
    if len(result) == 0:
        status = False
    return status

def get_emb_result(emb):
    if st.session_state['data'] == "DNP":
        query = f"""
        MATCH (n) WHERE 'Article' IN labels(n) OR 'Query' IN labels(n)
        RETURN n.name AS name, n.{emb} AS emb, left(n.name,1) AS category
        """      
    elif st.session_state['data'].startswith("P100") or st.session_state['data'].startswith("FP100"):
        query = f"""
        MATCH (n) WHERE 'Article' IN labels(n) OR 'Query' IN labels(n)
        RETURN n.name AS name, n.{emb} AS emb, labels(n)[0] AS category
        """  
    elif st.session_state['data'] == "euro_roads":
        query = f"""
        MATCH (p:Place)-[:IN_COUNTRY]->(country)
        WHERE country.code IN ["E", "GB", "F", "TR", "I", "D", "GR"]
        RETURN p.name AS name, p.{emb} AS emb, country.code AS category
        """  
    elif st.session_state['data'] == "newfood":   
        query = f""" 
        MATCH (dt:DishType)<-[:DISH_TYPE]-(d:Dish)
        WHERE dt.name IN ["bread", "drink", "salad", "sauce", "snack", "soup"]
        RETURN d.title as name, d.{emb} AS emb, dt.name AS category
        """
    elif st.session_state['data'] == "blogcatalog":   
        query = f""" 
        MATCH (n:Node)
        RETURN n.name as name, n.{emb} AS emb, labels(n)[0] AS category
        """  
    elif st.session_state['data'].startswith("airport"):   
        query = f""" 
        MATCH (n:Node)
        WHERE n.labels[0] IN ["1", "2", "3", "4", "5", "6", "7", "8", "9"]
        RETURN n.name as name, n.{emb} AS emb, n.labels[0] AS category
        """ 
    else:
        st.error("No embedding data is loaded!")
        st.stop()

    result = run(query)

    return result
