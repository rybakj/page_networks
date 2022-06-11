import os
import re
from datetime import datetime
from pathlib import Path
from urllib import request

import networkx as nx
import numpy as np
import pandas as pd
from bs4 import BeautifulSoup, SoupStrainer
from bs4.element import Doctype
from google.cloud import bigquery

from src.make_data.make_topology_matrix import (
    create_topology_matrix_pd,
    process_page_links,
)


def identify_seed_pages(seed0_pages):
    """
    Identifies seed pages used to create a functional network
        - Removes links to cross-domain services / external domain
        - Attaches anchor to the main url (/business#content)
        - Keeps the step-by-step ID, and search parameters
        - Only keeps self-loops where a page contains an explicit link to itself
        - Footer pages are manually defined and removed from the final list of seed
          pages. This is because they occur on every page, regardless of the seed0_pages
        - Code is adapted from: https://github.com/alphagov/govuk-intent-
          detector/blob/main/notebooks/generate_topology_matrix.ipynb

    Args:
        - seed0_pages: a list of GOVUK URL slugs that are considered vital to the Whole
          User Journey e.g.
                    ['/government/collections/ip-enforcement-reports',
                     '/government/publications/annual-ip-crime-and-enforcement-report-2020-to-2021',
                     '/search-registered-design']

    Returns:
        - A list of pages that are hyperlinked from seed0_pages.
    """

    # set up folders
    DIR_DATA_RAW = os.getenv("DIR_DATA_RAW")
    HTML_DATETIME = datetime.now().strftime("%Y%m%d_%H%M%S")
    DIR_HTML = Path(DIR_DATA_RAW, "html", HTML_DATETIME)
    DIR_HTML.mkdir(parents=True)

    # initialise an empty dictionary to store the HTML pages
    html_pages = {}

    # iterate over the requested GOV.UK pages
    for page in seed0_pages:

        # raise an error if the page doesn't start with a "/" - prevents the request
        # from hanging
        if not page.startswith("/"):
            raise ValueError(f"Pages must start with '/': {page}")

        # download the HTML page, store it in `html_pages`
        with request.urlopen(f"https://www.gov.uk{page}") as hp:
            html_page = hp.read().decode("utf8")

        # check if there is an anchor heading in the page URL; if there is one, only get
        # all the HTML **after** the anchor
        anchor_heading = re.match(r".*#(?P<anchor>[^/]+)$", page, flags=re.DOTALL)
        if anchor_heading:
            heading_string = str(
                BeautifulSoup(html_page).find(id=anchor_heading.group("anchor"))
            )
            html_page = heading_string + html_page.split(heading_string)[1]

        # write `html_page` out, replacing "/" with "__", as "/" is not a valid file
        # name, and also save it in a dictionary for further analysis
        with open(Path(DIR_HTML, f"{page.replace('/', '__')}.html"), "w") as f:
            _ = f.write(html_page)
        html_pages[page] = html_page

    # initialise an empty dictionary, and extract out the embedding hyperlinks in all
    # the HTML pages the HTML <a> tag defines a hyperlink. It has the following syntax:
    # <a href="url">link text</a>
    page_links = {}

    # iterate over the HTML files
    for html_page, html_contents in html_pages.items():

        # extract all embedded hyperlinks and save them in a list
        links = BeautifulSoup(html_contents, parse_only=SoupStrainer("a"))
        page_links[html_page] = [
            link.get("href") for link in links if not isinstance(link, Doctype)
        ]

    # process hyperlinks embedded in each page
    page_links_proc = process_page_links(page_links)

    # generate directed topology matrix, as a pandas.DataFrame
    topology_matrix_df = create_topology_matrix_pd(page_links=page_links_proc)

    # create a pandas.DataFrame of seed1 pages
    df = pd.DataFrame(
        topology_matrix_df.columns.values.tolist(), columns=["seed1_page"]
    )

    # remove footer pages
    footer_pages = [
        "/browse/disabilities",
        "/browse/housing-local-services",
        "/help",
        "/browse/tax",
        "/browse/childcare-parenting",
        "/",
        "/browse/employing-people",
        "/browse/environment-countryside",
        "/government/organisations/government-digital-service",
        "/help/terms-conditions",
        "/browse/benefits",
        "/help/cookies",
        "/browse/births-deaths-marriages",
        "/browse/abroad",
        "/coronavirus",
        "/contact",
        "/transition",
        "/government/how-government-works",
        "/browse/education",
        "/browse/justice",
        "/browse/citizenship",
        "/cymraeg",
        "/browse/working",
        "/browse/business",
        "/help/accessibility-statement",
        "/world",
        "/browse/visas-immigration",
        "/browse/driving",
        "/help/privacy-notice",
        "/government/organisations",
    ]

    return df[~df.seed1_page.isin(footer_pages)]["seed1_page"].values.tolist()


def extract_seed_sessions(start_date, end_date, seed0_pages, seed1_pages):
    """
    Retrieves all page hits from sessions that visit at least one seed0 or seed1
    page from google BigQuery.

        - URL parameteres/anchors are removed from the page paths, as we are only
          interested in the the gov.uk page path.
        - Seed1 pages found in the footer of the page are removed
        - Certain document types are ignored, see `documentTypesToIgnore`
        - Only page hits are included
        - Print pages are not included
        - Truncate URLs of 'simple_smart_answer', 'local_transaction', 'special_route',
          'licence', 'transaction', and 'Extension' document types, e.g.
          "/claim-tax-refund/y" TO "/claim-tax-refund". This is because we are not
          interested in the user's response, only the page.

    Args:
       - start_date: the start date for the session hit data
       - end_date: the end date for the session hit data
       - seed0_pages: a list of GOVUK URL slugs that are considered vital to the Whole
                      User Journey e.g. ['/browse/working', '/browse/disabilities',
                                         '/browse/tax']
       - seed1_pages: a list of GOVUK URL slugs that are hyperlinked from seed0_pages.
                      Get this list by calling `identify_seed_pages(seed0_pages)`.

    Returns:
       - A pd.DataFrame containing all page hit session data that has visited at least
         one seed0_page or seed1_page. `sessionId`, `hitNumber`, `pagePath`,
         `documentType`, `topLevelTaxons`, `bottomLevelTaxons`, `isEntrance`,
         `isExit` and total session hits for the pagePath are returned.
    """

    client = bigquery.Client(project="govuk-bigquery-analytics", location="EU")

    query = """
            DECLARE documentTypesToIgnore ARRAY <STRING>;

            SET documentTypesToIgnore = ['authored_article',
                                            'news_article',
                                            'news_story',
                                            'press_release',
                                            'world_news_story',
                                            'utaac_decision',
                                            'speech',
                                            'case_study',
                                            'raib_report',
                                            'asylum_support_decision',
                                            'policy_paper',
                                            'corporate_report',
                                            'written_statement',
                                            'consultation_outcome',
                                            'closed_consultation',
                                            'maib_report',
                                            'person',
                                            'correspondence',
                                            'employment_tribunal_decision',
                                            'employment_appeal_tribunal_decision',
                                            'tax_tribunal_decision',
                                            'ministerial_role',
                                            'residential_property_tribunal_decision',
                                            'cma_case',
                                            'completed_transaction',
                                            'Extension'];
            WITH primary_data AS (
                SELECT
                    hits.hitNumber,
                    REGEXP_REPLACE(hits.page.pagePath, r'[?#].*', '') AS pagePath,
                    CONCAT(fullVisitorId, "-", CAST(visitId AS STRING)) AS sessionId,
                    (SELECT value FROM hits.customDimensions WHERE index = 2)
                        AS documentType,
                    (SELECT value FROM hits.customDimensions WHERE index = 3)
                        AS topLevelTaxons,
                    (SELECT value FROM hits.customDimensions WHERE index = 58)
                        AS bottomLevelTaxons,
                    hits.isEntrance,
                    hits.isExit
                FROM `govuk-bigquery-analytics.87773428.ga_sessions_*`
                CROSS JOIN UNNEST(hits) AS hits
                WHERE
                    _TABLE_SUFFIX BETWEEN @startDate AND @endDate
                    AND hits.page.pagePath NOT LIKE "/print%"
                    AND hits.type = 'PAGE'
            ),

            -- remove irrelevant document types `documentTypesToIgnore`
            sessions_remove_document_types AS (
                SELECT
                    *
                FROM primary_data
                WHERE documentType NOT IN UNNEST(documentTypesToIgnore)
                    OR documentType IS NULL
            ),

              -- truncate URLs of certain document types
            sessions_truncate_urls AS (
                SELECT * REPLACE (
                    CASE
                        WHEN documentType IN ('smart_answer', 'simple_smart_answer',
                        'local_transaction', 'special_route', 'licence', 'transaction',
                        'Extension')
                        THEN REGEXP_EXTRACT(pagePath, r"^\\/[^\\/]+")
                        ELSE pagePath
                    END AS pagePath
                )
                FROM sessions_remove_document_types
            ),

            -- sessions which visit at least one `seed0_pages` or `seed1_pages`
            sessions_with_seed_0_or_1 AS (
                SELECT DISTINCT
                    sessionId
                FROM sessions_truncate_urls
                WHERE pagePath IN UNNEST(@seed0Pages)
                    OR pagePath IN UNNEST(@seed1Pages)
                GROUP BY sessionId, pagePath
            ),

            -- all session data (page hits) that visit at least one `seed0_pages` or
            -- `seed1_pages`
            all_sessions_seed_0_or_1 AS (
                SELECT
                    sessionId,
                    hitNumber,
                    pagePath,
                    documentType,
                    topLevelTaxons,
                    bottomLevelTaxons,
                    isEntrance,
                    isExit
                FROM sessions_truncate_urls
                WHERE sessionId IN (SELECT sessionId FROM sessions_with_seed_0_or_1)
                ORDER BY sessionId, hitNumber
            ),

            -- total session hits
            session_hits AS (
                SELECT
                    pagePath,
                    COUNT(DISTINCT sessionId) AS sessionHits
                FROM primary_data
                GROUP BY pagePath
            )

            -- join `session_hits` with `all_sessions_seed_0_or_1`
            SELECT
                all_sessions_seed_0_or_1.*,
                sessionHits
            FROM all_sessions_seed_0_or_1
            LEFT JOIN session_hits
            ON all_sessions_seed_0_or_1.pagePath = session_hits.pagePath
    """

    query_parameters = [
        bigquery.ScalarQueryParameter("startDate", "STRING", start_date),
        bigquery.ScalarQueryParameter("endDate", "STRING", end_date),
        bigquery.ArrayQueryParameter("seed0Pages", "STRING", seed0_pages),
        bigquery.ArrayQueryParameter("seed1Pages", "STRING", seed1_pages),
    ]

    return client.query(
        query, job_config=bigquery.QueryJobConfig(query_parameters=query_parameters)
    ).to_dataframe()


def extract_nodes_and_edges(page_view_network):
    """
    Extracts nodes and edges from a functional network.

    Args:
        - page_view_network: all page hit data from sessions that visit at least one
          seed0_page or seed1_page. Created via the function `extract_seed_sessions()`

    Returns:
        - nodes: pd.DataFrame with the node `sourcePagePath`, and the node properties
          `documentType`, `topLevelTaxons`, `bottomLevelTaxons`,
          `sourcePageSessionHitsAll`, `sourcePageSessionEntranceOnly`,
          `sourcePageSessionExitOnly`, `sourcePageSessionEntranceAndExit`, `sessionHits`
        - edges: pd.DataFrame with the edges `sourcePagePath` to `destinationPagePath`
          and the weight = `edgeWeight`. The edge weight `edgeWeight` is the number of
          distinct sessions that move between Page A and Page B.
    """

    df = page_view_network

    # rename pagePath to sourcePagePath
    df = df.rename(columns={"pagePath": "sourcePagePath"})

    # Create edges
    # find the source and destination pagePath
    df["destinationPagePath"] = (
        df.sort_values(by=["hitNumber"], ascending=True)
        .groupby(["sessionId"])["sourcePagePath"]
        .shift(-1)
    )

    df1 = df[["sessionId", "sourcePagePath", "destinationPagePath"]]

    # count the number of times a user session visits sourcePagePath to
    # destinationPagePath
    df1 = (
        df.groupby(["sourcePagePath", "destinationPagePath"], dropna=False)
        .sessionId.nunique()
        .reset_index(name="edgeWeight")
        .sort_values(by=["edgeWeight"], ascending=False)
    )

    # remove rows where pagePath = destinationPagePath
    df1[df1["sourcePagePath"] != df1["destinationPagePath"]]

    edges = df1

    # Create nodes
    df2 = df
    df2["isEntrance"] = df2["isEntrance"].astype("str")
    df2["isExit"] = df2["isExit"].astype("str")
    df2 = (
        df2.groupby(
            [
                "sourcePagePath",
                "documentType",
                "topLevelTaxons",
                "bottomLevelTaxons",
                "sessionHits",
                "isEntrance",
                "isExit",
            ],
            dropna=False,
        )
        .sessionId.nunique()
        .reset_index(name="counts")
    )

    # create unique columns depending on `counts`
    df2["sourcePageSessionHitsEntranceAndExit"] = np.where(
        (df2["isEntrance"] == "None") & (df2["isExit"] == "None"), df2["counts"], ""
    )

    df2["sourcePageSessionHitsEntranceOnly"] = np.where(
        (df2["isEntrance"] == "True") & (df2["isExit"] == "None"), df2["counts"], ""
    )

    df2["sourcePageSessionHitsExitOnly"] = np.where(
        (df2["isEntrance"] == "None") & (df2["isExit"] == "True"), df2["counts"], ""
    )

    df3 = (
        df2.groupby(
            [
                "sourcePagePath",
                "documentType",
                "topLevelTaxons",
                "bottomLevelTaxons",
                "sessionHits",
            ],
            dropna=False,
        )
        .agg(sourcePageSessionHitsAll=("counts", "sum"))
        .reset_index()
    )

    # merge df2 and df3
    df4 = pd.merge(
        df2,
        df3,
        how="left",
        on=[
            "sourcePagePath",
            "documentType",
            "topLevelTaxons",
            "bottomLevelTaxons",
            "sessionHits",
        ],
    )
    df4 = df4.drop(columns=["counts", "isEntrance", "isExit"])

    # fill nans and empty cells
    cols = [
        "sessionHits",
        "sourcePageSessionHitsAll",
        "sourcePageSessionHitsEntranceAndExit",
        "sourcePageSessionHitsEntranceOnly",
        "sourcePageSessionHitsExitOnly",
    ]
    df4[cols] = df4[cols].fillna(0)
    df4 = df4.fillna("no value")

    # collapse rows so there are no duplicated rows with blank cells
    df4 = (
        df4.mask(df4.astype(str).eq(""))
        .groupby(
            [
                "sourcePagePath",
                "documentType",
                "topLevelTaxons",
                "bottomLevelTaxons",
                "sessionHits",
                "sourcePageSessionHitsAll",
            ],
            dropna=False,
            as_index=False,
        )
        .first()
    )

    # select the top page path with document type, taxon, and session hit data. This is
    # because sometimes the tracking for taxons is incorrect, which results in multiple
    # rows for the same page path
    df4["RN"] = (
        df4.sort_values(
            ["sourcePagePath", "sourcePageSessionHitsAll"], ascending=[False, False]
        )
        .groupby(["sourcePagePath"], dropna=False)
        .cumcount()
        + 1
    )

    nodes = df4[df4["RN"] == 1]

    return (nodes, edges)


def create_networkx_graph(nodes, edges):
    """
    Combines the nodes and edges to create a NetworkX functional graph related to a
    set of seed pages.

    Args:
         - nodes: pd.DataFrame with the node `sourcePagePath`, and the node properties
          `documentType`, `topLevelTaxons`, `bottomLevelTaxons`,
          `sourcePageSessionHitsAll`, `sourcePageSessionEntranceOnly`,
          `sourcePageSessionExitOnly`, `sourcePageSessionEntranceAndExit`,
          `sessionHits`. Created via the function `extract_nodes_and_edges()`
        - edges: pd.DataFrame with the edges `sourcePagePath` to `destinationPagePath`
          and the weight = `edgeWeight`. The edge weight `edgeWeight` is the number of
          distinct sessions that move between Page A and Page B.  Created with the
          function `extract_nodes_and_edges()`

    Returns:
         - A NetworkX graph `G`

    """
    # add nodes, edges, and edge weight
    G = nx.from_pandas_edgelist(
        edges,
        "sourcePagePath",
        "destinationPagePath",
        ["edgeWeight"],
        create_using=nx.DiGraph(),
    )

    # iterate over nodes and set the source nodes' attributes
    nx.set_node_attributes(
        G,
        pd.Series(nodes.documentType.values, index=nodes.sourcePagePath).to_dict(),
        "documentType",
    )
    nx.set_node_attributes(
        G,
        pd.Series(nodes.topLevelTaxons.values, index=nodes.sourcePagePath).to_dict(),
        "topLevelTaxons",
    )
    nx.set_node_attributes(
        G,
        pd.Series(nodes.bottomLevelTaxons.values, index=nodes.sourcePagePath).to_dict(),
        "bottomLevelTaxons",
    )
    nx.set_node_attributes(
        G,
        pd.Series(
            nodes.sourcePageSessionHitsAll.values, index=nodes.sourcePagePath
        ).to_dict(),
        "sessionHitsAll",
    )
    nx.set_node_attributes(
        G,
        pd.Series(
            nodes.sourcePageSessionHitsEntranceOnly.values, index=nodes.sourcePagePath
        ).to_dict(),
        "entranceHit",
    )
    nx.set_node_attributes(
        G,
        pd.Series(
            nodes.sourcePageSessionHitsExitOnly.values, index=nodes.sourcePagePath
        ).to_dict(),
        "exitHit",
    )
    nx.set_node_attributes(
        G,
        pd.Series(
            nodes.sourcePageSessionHitsEntranceAndExit.values,
            index=nodes.sourcePagePath,
        ).to_dict(),
        "entranceAndExitHit",
    )
    nx.set_node_attributes(
        G,
        pd.Series(nodes.sessionHits.values, index=nodes.sourcePagePath).to_dict(),
        "sessionHits",
    )

    # remove nan nodes
    nan_nodes = []
    for node in G.nodes():
        if pd.isnull(node):
            nan_nodes.append(node)
    G.remove_nodes_from(nan_nodes)

    return G
