import os
import math
import json
import pandas as pd
from neo4j import GraphDatabase
import uuid

# =========================
# CONFIG
# =========================

NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "your_password"

IMPORT_DIR = "neo4j/import"  # change if needed

# =========================
# HELPERS
# =========================


def clean(x):
    if x is None:
        return None
    if isinstance(x, float) and math.isnan(x):
        return None
    return x


def clean_list(x):
    if clean(x) is None:
        return []
    return list(x)


def format_mathlib_code(gm):
    return gm.get("code", "")


# =========================
# EXPORT CSV FILES
# =========================


def export_csv(df):

    nodes = []
    sources = []
    contains = []
    object_refs = []
    entity_refs = []
    formal_nodes = []
    formalizes = []
    proves = []
    references = []

    for _, r in df.iterrows():

        uuid = r.uuid
        label = r.type.capitalize()

        # main node
        nodes.append(
            {
                "uuid": uuid,
                "label": label,
                "identifier": clean(r.identifier),
                "names": json.dumps(clean_list(r.names)),
                "text": clean(r.text),
                "variables": json.dumps(clean_list(r.local_variable_references)),
            }
        )

        # source
        sources.append(
            {
                "file_id": r.file_id,
                "uuid": uuid,
                "start": r.item_start,
                "end": r.item_end,
            }
        )

        # object refs (NO deduplication)
        for ref in clean_list(r.object_references):

            ref_uuid = os.urandom(16).hex()

            object_refs.append({"uuid": ref_uuid, "text": ref})

            contains.append({"parent_uuid": uuid, "child_uuid": ref_uuid})

        # entity refs
        for ref in clean_list(r.entity_references):

            ref_uuid = os.urandom(16).hex()

            entity_refs.append({"uuid": ref_uuid, "text": ref})

            contains.append({"parent_uuid": uuid, "child_uuid": ref_uuid})

        # formal nodes
        if r.type == "definition":

            names = clean_list(r.names)
            links = clean_list(r.mathlib_links)

            for name, link in zip(names, links):

                if not link:
                    continue

                gm = link.get("grounded_match")

                if not gm:
                    continue

                f_uuid = uuid.uuid4()

                formal_nodes.append(
                    {
                        "uuid": f_uuid,
                        "informal_name": name,
                        "formal_name": ".".join(gm["name"]),
                        "module": ".".join(gm["module_name"]),
                        "code": format_mathlib_code(gm),
                        "source": "mathlib",
                        "status": "compiling",
                    }
                )

                formalizes.append({"formal_uuid": f_uuid, "parent_uuid": uuid})

        # proves
        if clean(r.proof_link):

            proves.append({"proof_uuid": r.proof_link, "target_uuid": uuid})

        # references
        for dst in clean_list(r.object_links):

            references.append({"src": uuid, "dst": dst})

        for dst in clean_list(r.entity_links):

            references.append({"src": uuid, "dst": dst})

    # write CSVs
    pd.DataFrame(nodes).to_csv(f"{IMPORT_DIR}/nodes.csv", index=False)
    pd.DataFrame(sources).to_csv(f"{IMPORT_DIR}/sources.csv", index=False)
    pd.DataFrame(object_refs).to_csv(f"{IMPORT_DIR}/object_refs.csv", index=False)
    pd.DataFrame(entity_refs).to_csv(f"{IMPORT_DIR}/entity_refs.csv", index=False)
    pd.DataFrame(contains).to_csv(f"{IMPORT_DIR}/contains.csv", index=False)
    pd.DataFrame(formal_nodes).to_csv(f"{IMPORT_DIR}/formal_nodes.csv", index=False)
    pd.DataFrame(formalizes).to_csv(f"{IMPORT_DIR}/formalizes.csv", index=False)
    pd.DataFrame(proves).to_csv(f"{IMPORT_DIR}/proves.csv", index=False)
    pd.DataFrame(references).to_csv(f"{IMPORT_DIR}/references.csv", index=False)


# =========================
# LOAD CSV INTO NEO4J
# =========================


def load_csv():

    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

    with driver.session() as session:

        # indexes
        session.run("CREATE INDEX node_uuid IF NOT EXISTS FOR (n:Definition) ON n.uuid")
        session.run("CREATE INDEX node_uuid IF NOT EXISTS FOR (n:Theorem) ON n.uuid")
        session.run("CREATE INDEX node_uuid IF NOT EXISTS FOR (n:Exercise) ON n.uuid")
        session.run("CREATE INDEX node_uuid IF NOT EXISTS FOR (n:Example) ON n.uuid")
        session.run("CREATE INDEX node_uuid IF NOT EXISTS FOR (n:Proof) ON n.uuid")
        session.run(
            "CREATE INDEX source_file_id IF NOT EXISTS FOR (s:Source) ON s.file_id"
        )

        # nodes
        session.run(
            """
        LOAD CSV WITH HEADERS FROM 'file:///nodes.csv' AS row

        CALL apoc.create.node(
            [row.label],
            {
                uuid: row.uuid,
                identifier: row.identifier,
                names: apoc.convert.fromJsonList(row.names),
                text: row.text,
                variables: apoc.convert.fromJsonList(row.variables)
            }
        ) YIELD node RETURN node;
        """
        )

        # sources
        session.run(
            """
        LOAD CSV WITH HEADERS FROM 'file:///sources.csv' AS row

        MATCH (n {uuid: row.uuid})

        MERGE (s:Source {file_id: row.file_id})

        CREATE (s)-[:Provides {
            start: toInteger(row.start),
            end: toInteger(row.end)
        }]->(n)
        """
        )

        # object refs
        session.run(
            """
        LOAD CSV WITH HEADERS FROM 'file:///object_refs.csv' AS row

        CREATE (:ObjectReference {
            uuid: row.uuid,
            text: row.text
        })
        """
        )

        # entity refs
        session.run(
            """
        LOAD CSV WITH HEADERS FROM 'file:///entity_refs.csv' AS row

        CREATE (:EntityReference {
            uuid: row.uuid,
            text: row.text
        })
        """
        )

        # contains
        session.run(
            """
        LOAD CSV WITH HEADERS FROM 'file:///contains.csv' AS row

        MATCH (p {uuid: row.parent_uuid})
        MATCH (c {uuid: row.child_uuid})

        CREATE (p)-[:Contains]->(c)
        """
        )

        # formal nodes
        session.run(
            """
        LOAD CSV WITH HEADERS FROM 'file:///formal_nodes.csv' AS row

        CREATE (:Formal {
            uuid: row.uuid,
            informal_name: row.informal_name,
            formal_name: row.formal_name,
            module: row.module,
            code: row.code,
            source: row.source,
            status: row.status
        })
        """
        )

        # formalizes
        session.run(
            """
        LOAD CSV WITH HEADERS FROM 'file:///formalizes.csv' AS row

        MATCH (f:Formal {uuid: row.formal_uuid})
        MATCH (p {uuid: row.parent_uuid})

        CREATE (f)-[:Formalizes]->(p)
        """
        )

        # proves
        session.run(
            """
        LOAD CSV WITH HEADERS FROM 'file:///proves.csv' AS row

        MATCH (p:Proof {uuid: row.proof_uuid})
        MATCH (t {uuid: row.target_uuid})

        CREATE (p)-[:Proves]->(t)
        """
        )

        # references
        session.run(
            """
        LOAD CSV WITH HEADERS FROM 'file:///references.csv' AS row

        MATCH (a {uuid: row.src})
        MATCH (b {uuid: row.dst})

        CREATE (a)-[:References]->(b)
        """
        )

    driver.close()


# =========================
# RUN
# =========================

if __name__ == "__main__":

    df = pd.read_json("sample_defs_grounded.json")
    export_csv(df)
    load_csv()
