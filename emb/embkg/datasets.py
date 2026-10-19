"""Dataset registry: what the app can load and how to read categories back."""

from embkg import paths

# type -> {dataset: caption}
DATASETS = {
    "DNP": {
        "DNP": "This database includes 100 DNP newsreleases, and 4 Toppan newsreleases.",
    },
    "WIKI": {
        "FP100": "This database includes wikipedia pages of 100 football players.",
        "P100": "This database includes wikipedia pages of 100 persons, consisting of 25 athletes, 25 engineers, 25 actors, and 25 politicians.",
        "P1000": "This database includes wikipedia pages of 1000 persons, consisting of 100 athletes, 100 engineers, 100 actors, 100 politicians, 100 physicians, 100 scientists, 100 artists, 100 journalists, 100 soldiers, and 100 lawyers.",
        "P10000": "This database includes wikipedia pages of 10000 persons, consisting of 1000 athletes, 1000 engineers, 1000 actors, 1000 politicians, 1000 physicians, 1000 scientists, 1000 artists, 1000 journalists, 1000 soldiers, and 1000 lawyers.",
    },
    "CYPHER": {
        "euro_roads": "The dataset contains 894 towns, 39 countries, and 1,250 roads connecting them. https://github.com/neo4j-examples/graph-embeddings/raw/main/data/roads.csv",
        "newfood": "The dataset contains nutritional information alongside the ingredients used in 1600+ dishes. https://raw.githubusercontent.com/smallcat9603/graph/main/data/newfood.csv",
        "blogcatalog": "The dataset contains undirected social network of bloggers with (multi) labels representing topics of interest. https://raw.githubusercontent.com/smallcat9603/graph/main/data/blogcatalog_0.edges",
        "airport": "The dataset contains flight network among global airports (from Open-Flights). https://raw.githubusercontent.com/smallcat9603/graph/main/data/airport_0.edges",
        "airport0.8": "The dataset contains flight network among global airports (from Open-Flights), however 20 percent of edges are randomly removed while keeping graph connected. https://raw.githubusercontent.com/smallcat9603/graph/main/data/airport_0_remaining.edges",
    },
}

LANGUAGE = {"DNP": "ja", "WIKI": "en", "CYPHER": "en"}

CYPHER_FILES = {
    "euro_roads": paths.CYPHER_URL + "euro_roads.cypher",
    "newfood": paths.CYPHER_URL + "newfood.cypher",
    "blogcatalog": paths.CYPHER_URL + "blogcatalog.cypher",
    "airport": paths.CYPHER_URL + "airport.cypher",
    "airport0.8": paths.CYPHER_URL + "airport_removed.cypher",
}

WIKI_CSV = {
    "FP100": "wikidata_footballplayer_100.csv",
    "P100": "wikidata_persons_100.csv",
    "P1000": "wikidata_persons_1000.csv",
    "P10000": "wikidata_persons_10000.csv",
}

DNP_QUERY_URLS = {
    "C-1": "https://www.holdings.toppan.com/ja/news/2023/10/newsrelease231004_1.html",
    "C-2": "https://www.holdings.toppan.com/ja/news/2023/10/newsrelease231004_2.html",
    "C-3": "https://www.holdings.toppan.com/ja/news/2023/10/newsrelease231004_3.html",
    "C-4": "https://www.holdings.toppan.com/ja/news/2023/10/newsrelease231003_1.html",
}


def dnp_source(data_type):
    """Return (data_url, query_dict) for the DNP dataset."""
    if data_type == "TXT":
        data_url = str(paths.NEWSRELEASE_DIR) + "/"
        query_dict = {q: f"{data_url}{q}.txt" for q in DNP_QUERY_URLS}
    else:
        data_url = f"{paths.KG_DATA_URL}articles.csv"
        query_dict = dict(DNP_QUERY_URLS)
    return data_url, query_dict


def wiki_source(data):
    """Return (data_url, query_dict) for a WIKI dataset."""
    data_url = f"{paths.KG_DATA_URL}{WIKI_CSV[data]}"
    if data == "FP100":
        query_dict = {"Thierry Henry": "https://en.wikipedia.org/wiki/Thierry_Henry"}
    else:
        query_dict = {"Joe Biden": "https://en.wikipedia.org/wiki/Joe_Biden"}
    return data_url, query_dict


def emb_result_query(data, emb):
    """Cypher returning (name, emb, category) for evaluating embedding `emb` on `data`.

    Returns None when the dataset has no category definition.
    """
    if data == "DNP":
        return f"""
        MATCH (n) WHERE 'Article' IN labels(n) OR 'Query' IN labels(n)
        RETURN n.name AS name, n.{emb} AS emb, left(n.name,1) AS category
        """
    if data.startswith("P100") or data.startswith("FP100"):
        return f"""
        MATCH (n) WHERE 'Article' IN labels(n) OR 'Query' IN labels(n)
        RETURN n.name AS name, n.{emb} AS emb, labels(n)[0] AS category
        """
    if data == "euro_roads":
        return f"""
        MATCH (p:Place)-[:IN_COUNTRY]->(country)
        WHERE country.code IN ["E", "GB", "F", "TR", "I", "D", "GR"]
        RETURN p.name AS name, p.{emb} AS emb, country.code AS category
        """
    if data == "newfood":
        return f"""
        MATCH (dt:DishType)<-[:DISH_TYPE]-(d:Dish)
        WHERE dt.name IN ["bread", "drink", "salad", "sauce", "snack", "soup"]
        RETURN d.title as name, d.{emb} AS emb, dt.name AS category
        """
    if data == "blogcatalog":
        return f"""
        MATCH (n:Node)
        RETURN n.name as name, n.{emb} AS emb, labels(n)[0] AS category
        """
    if data.startswith("airport"):
        return f"""
        MATCH (n:Node)
        WHERE n.labels[0] IN ["1", "2", "3", "4", "5", "6", "7", "8", "9"]
        RETURN n.name as name, n.{emb} AS emb, n.labels[0] AS category
        """
    return None
