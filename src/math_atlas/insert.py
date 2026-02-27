import csv
import json
import math
import os
import uuid

import pandas as pd

from neo4j import GraphDatabase

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
        label = r.type.capitalize()

        # main node
        nodes.append(
            {
                "uuid": r.uuid,
                "labels": [label, "Item"],
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
                "uuid": r.uuid,
                "start": r.item_start,
                "end": r.item_end,
            }
        )

        # object refs (NO deduplication)
        for ref, target in zip(clean_list(r.object_references), clean_list(r.object_links)):
            ref_uuid = uuid.uuid4()

            object_refs.append({"uuid": ref_uuid, "text": ref})

            contains.append({"parent_uuid": r.uuid, "child_uuid": ref_uuid})
            for tl in target:
                references.append({"src": ref_uuid, "dst": tl})

        # entity refs
        for ref, target in zip(clean_list(r.entity_references), clean_list(r.entity_links)):
            ref_uuid = uuid.uuid4()

            entity_refs.append({"uuid": ref_uuid, "text": ref})

            contains.append({"parent_uuid": r.uuid, "child_uuid": ref_uuid})
            references.append({"src": ref_uuid, "dst": target})

        # formal nodes
        if r.type == "definition":
            names = clean_list(r.names)
            links = clean_list(r.mathlib_links) if hasattr(r, "mathlib_links") else []

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

                formalizes.append({"formal_uuid": f_uuid, "parent_uuid": r.uuid})

        # proves
        if clean(r.proof_link):
            proves.append({"theorem_uuid": r.proof_link, "proof_uuid": r.uuid})

    def to_csv(df, path):
        df.to_csv(
            path,
            index=False,
            quoting=csv.QUOTE_ALL,  # quote every field
            escapechar="\\",  # escape internal quotes
            doublequote=True,
            lineterminator="\n",
        )

    # write CSVs
    to_csv(pd.DataFrame(nodes), f"{IMPORT_DIR}/nodes.csv")
    to_csv(pd.DataFrame(sources), f"{IMPORT_DIR}/sources.csv")
    to_csv(pd.DataFrame(object_refs), f"{IMPORT_DIR}/object_refs.csv")
    to_csv(pd.DataFrame(entity_refs), f"{IMPORT_DIR}/entity_refs.csv")
    to_csv(pd.DataFrame(contains), f"{IMPORT_DIR}/contains.csv")
    to_csv(pd.DataFrame(formal_nodes), f"{IMPORT_DIR}/formal_nodes.csv")
    to_csv(pd.DataFrame(formalizes), f"{IMPORT_DIR}/formalizes.csv")
    to_csv(pd.DataFrame(proves), f"{IMPORT_DIR}/proves.csv")
    to_csv(pd.DataFrame(references), f"{IMPORT_DIR}/references.csv")


# =========================
# LOAD CSV INTO NEO4J
# =========================


def load_csv():

    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

    with driver.session() as session:
        # indexes
        session.run("CREATE INDEX item_uuid IF NOT EXISTS FOR (n:Item) ON n.uuid")
        session.run("CREATE INDEX def_uuid IF NOT EXISTS FOR (n:Definition) ON n.uuid")
        session.run("CREATE INDEX thm_uuid IF NOT EXISTS FOR (n:Theorem) ON n.uuid")
        session.run("CREATE INDEX exc_uuid IF NOT EXISTS FOR (n:Exercise) ON n.uuid")
        session.run("CREATE INDEX exm_uuid IF NOT EXISTS FOR (n:Example) ON n.uuid")
        session.run("CREATE INDEX prf_uuid IF NOT EXISTS FOR (n:Proof) ON n.uuid")
        session.run("CREATE INDEX src_uuid IF NOT EXISTS FOR (s:Source) ON s.file_id")

        session.run("CREATE INDEX formal_uuid IF NOT EXISTS FOR (n:Formal) ON n.uuid")

        session.run("CREATE INDEX ref_uuid IF NOT EXISTS FOR (n:Reference) ON n.uuid")
        session.run("CREATE INDEX obj_uuid IF NOT EXISTS FOR (n:ObjectReference) ON n.uuid")
        session.run("CREATE INDEX ent_uuid IF NOT EXISTS FOR (n:EntityReference) ON n.uuid")

        # nodes
        session.run("""
        LOAD CSV WITH HEADERS FROM 'file:///nodes.csv' AS row

        CALL apoc.create.node(
            apoc.convert.fromJsonList(row.labels),
            {
                uuid: row.uuid,
                identifier: row.identifier,
                names: apoc.convert.fromJsonList(row.names),
                text: row.text,
                variables: apoc.convert.fromJsonList(row.variables)
            }
        ) YIELD node
        RETURN node;
        """)
        print("loaded nodes")

        # sources
        session.run("""
        LOAD CSV WITH HEADERS FROM 'file:///sources.csv' AS row

        MATCH (n:Item)
        WHERE n.uuid = row.uuid
        MERGE (s:Source {file_id: row.file_id})
        CREATE (s)-[:Provides {
            start: toInteger(row.start),
            end: toInteger(row.end)
        }]->(n)
        """)
        print("loaded sources")

        # object refs
        session.run("""
        LOAD CSV WITH HEADERS FROM 'file:///object_refs.csv' AS row

        CREATE (:ObjectReference:Reference {
            uuid: row.uuid,
            text: row.text
        })
        RETURN count(*) as count
        """)
        print("loaded object refs")

        # entity refs
        session.run("""
        LOAD CSV WITH HEADERS FROM 'file:///entity_refs.csv' AS row

        CREATE (:EntityReference:Reference {
            uuid: row.uuid,
            text: row.text
        })
        RETURN count(*) as count
        """)
        print("loaded entity refs")

        # contains
        session.run("""
        LOAD CSV WITH HEADERS FROM 'file:///contains.csv' AS row

        MATCH (p:Item {uuid: row.parent_uuid})
        MATCH (c:Reference {uuid: row.child_uuid})

        CREATE (p)-[:Contains]->(c)
        """)
        print("loaded contains")

        # formal nodes
        session.run("""
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
        """)
        print("loaded formals")

        # formalizes
        session.run("""
        LOAD CSV WITH HEADERS FROM 'file:///formalizes.csv' AS row

        MATCH (f:Formal {uuid: row.formal_uuid})
        MATCH (p:Item {uuid: row.parent_uuid})

        CREATE (f)-[:Formalizes]->(p)
        """)
        print("loaded formalizes")

        # proves
        session.run("""
        LOAD CSV WITH HEADERS FROM 'file:///proves.csv' AS row

        MATCH (p:Proof {uuid: row.proof_uuid})
        MATCH (t:Item {uuid: row.theorem_uuid})

        CREATE (p)-[:Proves]->(t)
        """)
        print("loaded proves")

        # references
        session.run("""
        LOAD CSV WITH HEADERS FROM 'file:///references.csv' AS row

        MATCH (a:Reference {uuid: row.src})
        MATCH (b:Item {uuid: row.dst})

        CREATE (a)-[:References]->(b)
        """)
        print("loaded references")

    driver.close()


# =========================
# RUN
# =========================

if __name__ == "__main__":
    df = pd.read_json("math_atlas_v4_grounded.json")
    export_csv(df)
    load_csv()
