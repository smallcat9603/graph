"""Cypher queries used to build, inspect and reset the knowledge graph.

Every function takes `run`, a callable query -> DataFrame (e.g. gds.run_cypher,
or the cached runner in the app).
"""

import re

import numpy
import pandas as pd

from embkg import nlp


def _numeric_properties(run, match, var, result):
    props = [value for value in result if isinstance(value, str) and not value.startswith("__csv_")]
    numeric = []
    for prop in props:
        value = run(f"{match} RETURN {var}.`{prop}` AS propertyValue")["propertyValue"][0]
        if isinstance(value, (float, int, numpy.int64)) or (isinstance(value, list) and isinstance(value[0], (float, int, numpy.int64))):
            numeric.append(prop)
    return numeric


def get_node_labels(run):
    return list(run("CALL db.labels()")["label"])


def get_node_properties(run, node_label):
    """Numeric properties of nodes with this label."""
    match = f"MATCH (n:{node_label})"
    result = run(f"{match} RETURN DISTINCT keys(n) AS properties")["properties"][0]
    return _numeric_properties(run, match, "n", result)


def get_relationship_types(run):
    return list(run("CALL db.relationshipTypes()")["relationshipType"])


def get_relationship_properties(run, rel):
    """Numeric properties of relationships of this type."""
    match = f"MATCH ()-[r:{rel}]->()"
    result = run(f"{match} RETURN DISTINCT keys(r) AS properties")["properties"][0]
    return _numeric_properties(run, match, "r", result)


def create_constraint(run, constraint_name):
    run(f"""
    CREATE CONSTRAINT {constraint_name} IF NOT EXISTS
    For (a:Article) REQUIRE a.url IS UNIQUE;
    """)


def post_process(run):
    """Convert string properties of an Offline (CSV) import back to numbers and lists."""
    run("""
    MATCH (n) WHERE n.pr0 IS NOT NULL
    SET n.pr0 = toFloat(n.pr0)
    """)
    run("""
    MATCH (n) WHERE n.pr1 IS NOT NULL
    SET n.pr1 = toFloat(n.pr1)
    SET n.pr2 = toFloat(n.pr2)
    SET n.pr3 = toFloat(n.pr3)
    """)
    run("""
    MATCH (n) WHERE n.phrase IS NOT NULL
    SET n.salience = apoc.convert.fromJsonList(n.salience)
    """)
    run("""
    MATCH ()-[r:CONTAINS]-() WHERE r.rank IS NOT NULL
    SET r.rank = toInteger(r.rank)
    SET r.weight = toInteger(r.weight)
    SET r.score = toFloat(r.score)
    """)
    run("""
    MATCH ()-[r]-() WHERE type(r) =~ 'SIMILAR_.*'
    SET r.score = toFloat(r.score)
    """)
    run("""
    MATCH (n) WHERE n.phrase IS NOT NULL
    SET n.phrase = replace(n.phrase, "[", "")
    SET n.phrase = replace(n.phrase, "]", "")
    SET n.phrase = split(n.phrase, ",")
    """)
    run("""
    MATCH ()-[r:CORRELATES]-() WHERE r.common IS NOT NULL
    SET r.common = replace(r.common, "[", "")
    SET r.common = replace(r.common, "]", "")
    WITH r, r.common AS common
    SET r.common = split(common, ",")
    """)


def get_nodes_relationships_csv(file):
    """Node labels and relationship types listed in an apoc.export.csv.all file."""
    df = pd.read_csv(file)

    nodes = [v for v in df["_labels"].unique().tolist() if isinstance(v, str)]
    nodes = [v[1:] if v.startswith(":") else v for v in nodes]

    relationships = [v for v in df["_type"].unique().tolist() if isinstance(v, str)]
    relationships = [v[1:] if v.startswith(":") else v for v in relationships]

    return nodes, relationships


def import_graph_data(run, data_dir_url, DATA):
    """Offline mode: bulk-import the exported node and relationship CSVs of DATA."""
    nodes, relationships = get_nodes_relationships_csv(f"{data_dir_url}{DATA}.csv")
    node_files = ", ".join(f"{{fileName: '{data_dir_url}{DATA}.nodes.{node}.csv', labels: ['{node}']}}" for node in nodes)
    rel_files = ", ".join(f"{{fileName: '{data_dir_url}{DATA}.relationships.{rel}.csv', type: '{rel}'}}" for rel in relationships)
    result = run(f"CALL apoc.import.csv([{node_files}], [{rel_files}], {{}})")
    post_process(run)
    return result


def load_articles_txt(run, data_dir, query_dict):
    """Create Article nodes B-1..B-100 and Query nodes from local text files."""
    for idx in range(1, 101):
        node = "B-" + str(idx)
        file = data_dir + node + ".txt"
        with open(file, 'r') as f:
            content = re.sub('\n+', ' ', f.read())
        run(f"""
        MERGE (a:Article {{ name: "{node}", url: "{file}", body: "{content}" }})
        """)
    for name, file in query_dict.items():
        with open(file, 'r') as f:
            content = re.sub('\n+', ' ', f.read())
        run(f"""
        MERGE (q:Query {{ name: "{name}", url: "{file}", body: "{content}" }})
        """)


def load_data_url(run, url):
    """Create Article nodes from a CSV of URLs and fetch their HTML body (APOC Extended)."""
    run(f"""
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
    """)


def create_query_node(run, name, url):
    """Create one Query node and fetch its HTML body (APOC Extended)."""
    run(f"""
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
    """)


def set_phrase_salience_properties_csv(run, file, query_node=False):
    """Semi-Online mode: read precomputed phrase/salience lists from CSV."""
    if not query_node:
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


def set_phrase_salience_properties_gcp(run, gcp_api_key, query_node=False):
    """Online mode: Google Cloud NLP entity extraction (APOC Extended)."""
    if not query_node:
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


def set_phrase_salience_properties_spacy(run, language, word_class, pipeline_size, n, query_node=False):
    """On-the-fly mode: top-n spaCy keywords per Article (or Query) node."""
    model = nlp.get_nlp(language, pipeline_size)
    label, var = ("Query", "q") if query_node else ("Article", "a")
    result = run(f"""
    MATCH ({var}:{label})
    RETURN {var}.url as url, {var}.body AS text
    """)

    for url, text in zip(result["url"], result["text"]):
        top_keywords = nlp.extract_keywords(model, text, word_class, n)
        phrase = [item[0] for item in top_keywords]
        salience = [item[1] for item in top_keywords]
        run(f"""
        MATCH ({var}:{label} {{url: "{url}"}})
        SET {var}.processed = true
        SET {var}.phrase = {phrase}
        SET {var}.salience = {salience}
        """)

    return result


def create_noun_article_relationships(run, nphrase, query_node=False):
    """(Article|Query)-[:CONTAINS {rank, score, weight}]-(Noun) for the top nphrase phrases."""
    label, var = ("Query", "q") if query_node else ("Article", "a")
    run(f"""
    MATCH ({var}:{label})
    WHERE {var}.processed IS NOT NULL
    FOREACH (word IN {var}.phrase[0..{nphrase}] |
    MERGE (n:Noun {{name: word}})
    MERGE ({var})-[r:CONTAINS]-(n)
    SET r.rank = apoc.coll.indexOf({var}.phrase, word) + 1
    SET r.score = {var}.salience[apoc.coll.indexOf({var}.phrase, word)]
    SET r.weight = {nphrase} - apoc.coll.indexOf({var}.phrase, word)
    )
    """)


def create_article_article_relationships(run, nphrase, query_node=False):
    """CORRELATES edges between nodes sharing at least one top-nphrase phrase."""
    if not query_node:
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


def get_graph_statistics(run):
    return run("""
    CALL apoc.meta.stats()
    YIELD nodeCount, relCount, labels, relTypesCount
    RETURN nodeCount, relCount, labels, relTypesCount
    """)


def save_graph_data(run, data):
    """Export the whole graph to <neo4j import dir>/<data>.csv (all-in-one and bulkImport)."""
    result_allinone = run(f"""
    CALL apoc.export.csv.all("{data}.csv", {{}})
    """)
    run(f"""
    CALL apoc.export.csv.all("{data}.csv", {{bulkImport: true}})
    """)
    return result_allinone


def free_up_db(run):
    """Delete all nodes and drop all constraints."""
    run("MATCH (n) DETACH DELETE n")
    for constraint in run("SHOW CONSTRAINTS YIELD name RETURN name")["name"]:
        run(f"DROP CONSTRAINT {constraint} IF EXISTS")


def run_file(run, file_cypher):
    """Run a .cypher file from a URL (APOC Extended)."""
    run(f'CALL apoc.cypher.runFile("{file_cypher}")')


def get_emb_table(run):
    return run("""
    MATCH (n)
    RETURN n.name AS name, n.emb_frp AS emb_frp, n.emb_n2v AS emb_n2v
    """)


def drop_emb(run):
    run("MATCH (n) REMOVE n.emb_frp")
    run("MATCH (n) REMOVE n.emb_n2v")


def emb_exists(run, emb):
    return len(run(f"MATCH (n) WHERE n.{emb} IS NOT NULL RETURN n")) > 0


# Read-only queries behind the "UI Interaction" tabs.

def knn_query(emb, query_node, limit):
    m = {"emb_frp": "FASTRP", "emb_n2v": "NODE2VEC", "emb_hgn": "HASHGNN"}[emb]
    return f"""
    MATCH (q:Query)-[r:SIMILAR_{m}]-(a:Article) WHERE q.name = "{query_node}"
    RETURN q.name AS Query, a.name AS Article, a.url AS URL, a.grp AS Group, a.grp1 AS Group1, r.score AS Similarity
    ORDER BY Similarity DESC
    LIMIT {limit}
    """


def node_similarity_query(query_node, similarity_method, limit):
    if similarity_method == "PPR":
        return f"""
        MATCH (q:Query)-[r:CORRELATES]-(a:Article) WHERE q.name = "{query_node}"
        RETURN q.name AS Query, a.name AS Article, a.url AS URL, a.pr{str(int(query_node.split("-")[-1])-1)} AS Similarity
        ORDER BY Similarity DESC
        LIMIT {limit}
        """
    return f"""
    MATCH (q:Query)-[r:SIMILAR_{similarity_method}]-(a:Article) WHERE q.name = "{query_node}"
    RETURN q.name AS Query, a.name AS Article, a.url AS URL, r.score AS Similarity
    ORDER BY Similarity DESC
    LIMIT {limit}
    """


def multiple_queries_query(query_nodes, similarity_method, limit):
    if similarity_method == "PPR":
        ppr_attr = ["pr" + str(int(item.replace("C-", ""))-1) for item in query_nodes]
        return f"""
        MATCH (q:Query)-[r:CORRELATES]-(a:Article) WHERE q.name IN {query_nodes}
        RETURN COLLECT(q.name) AS Query, a.name AS Article, REDUCE(s = 0, pr IN {ppr_attr} | s + a[pr]) AS Similarity
        ORDER BY Similarity DESC
        LIMIT {limit}
        """
    return f"""
    MATCH (q:Query)-[r:SIMILAR_{similarity_method}]-(a:Article) WHERE q.name IN {query_nodes}
    RETURN COLLECT(q.name) AS Query, a.name AS Article, SUM(r.score) AS Similarity
    ORDER BY Similarity DESC
    LIMIT {limit}
    """


def related_articles_query(noun):
    return f"""
    MATCH (n:Noun)-[]-(a:Article) WHERE n.name CONTAINS "{noun}"
    WITH DISTINCT a AS distinctArticle, n
    RETURN n.name AS Keyword, COUNT(distinctArticle) AS articleCount, COLLECT(distinctArticle.name) AS articles
    ORDER BY articleCount DESC
    """


COMMON_KEYWORDS_QUERY = """
    MATCH (n:Noun)-[]-(a:Article)
    RETURN n.name AS Keyword, COUNT(a) AS articleCount, COLLECT(a.name) AS articles
    ORDER BY articleCount DESC
    """

NAIVE_BY_RANK_QUERY = """
    MATCH (q:Query)-[r:CONTAINS]-(n:Noun)-[c:CONTAINS]-(a:Article)
    RETURN q.name AS Query, a.name AS Article, a.url AS URL, a.grp AS Group, a.grp1 AS Group1, collect(n.name) AS Common, SUM((1.0/r.rank)*(1.0/c.rank)) AS Similarity
    ORDER BY Query, Similarity DESC
    LIMIT 10
    """

NAIVE_BY_SALIENCE_QUERY = """
    MATCH (q:Query)-[r:CORRELATES]-(a:Article)
    WITH q, r, a, reduce(s = 0.0, word IN r.common |
    s + q.salience[apoc.coll.indexOf(q.phrase, word)] + a.salience[apoc.coll.indexOf(a.phrase, word)]) AS Similarity
    RETURN q.name AS Query, a.name AS Article, a.url AS URL, a.grp AS Group, a.grp1 AS Group1, r.common, Similarity
    ORDER BY Query, Similarity DESC
    LIMIT 10
    """
