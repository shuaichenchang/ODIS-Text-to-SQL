import os
import json
import copy
import random
import numpy as np
from numpy.linalg import norm
from collections import defaultdict
import sqlparse
from rank_bm25 import BM25Okapi

spider_dev_db_ids = [
    'concert_singer',
    'pets_1',
    'car_1',
    'flight_2',
    'employee_hire_evaluation',
    'cre_Doc_Template_Mgt',
    'course_teach',
    'museum_visit',
    'wta_1',
    'battle_death',
    'student_transcripts_tracking',
    'tvshow',
    'poker_player',
    'voter_1',
    'world_1',
    'orchestra',
    'network_1',
    'dog_kennels',
    'singer',
    'real_estate_properties'
]

CLAUSE_KEYWORDS = ['select', 'from', 'where', 'group by', 'order by', 'limit', 'intersect', 'union', 'except']
JOIN_KEYWORDS = ['join', 'on', 'as']
WHERE_OPS = ['not', 'between', 'in', 'like', 'is', 'exists', '=', '>', '<', '>=', '<=', '!=']
UNIT_OPS = ['-', '+']
AGG_OPS = ['max', 'min', 'count', 'sum', 'avg']
COND_OPS = ['and', 'or']
ORDER_OPS = ['desc', 'asc']
SQL_KEYWORDS = []
SQL_KEYWORDS.extend(CLAUSE_KEYWORDS)
SQL_KEYWORDS.extend(JOIN_KEYWORDS)
SQL_KEYWORDS.extend(WHERE_OPS)
SQL_KEYWORDS.extend(UNIT_OPS)
SQL_KEYWORDS.extend(AGG_OPS)
SQL_KEYWORDS.extend(COND_OPS)
SQL_KEYWORDS.extend(ORDER_OPS)


def format_query(q, format_type):
    if format_type == 'original':
        return q["query"]
    elif format_type == 'normalized':
        return q["gold"]["query_normalized"]
    else:
        raise ValueError(f"format_type {format_type} not supported")


def lexical(query, values):
    if isinstance(query, str):
        for placeholder, value in values.items():
            query = query.replace(placeholder, value)
    elif isinstance(query, list):
        for i in range(len(query)):
            if query[i] in values:
                query[i] = values[query[i]]
    return query


def delexical(query):
    values = {}
    new_query = ""
    in_value = False
    in_col = False
    value = ""
    placeholder_id = 0
    new_query = ""
    for char in query:
        if char == "'":
            in_value = not in_value
            value += char
            if not in_value:
                values[f"value_{placeholder_id}"] = value
                new_query += f"value_{placeholder_id}"
                placeholder_id += 1
                value = ""
        else:
            if not in_value:
                new_query += char
            else:
                value += char
    return new_query, values


def _is_whitespace(sqlparse_token):
    return sqlparse_token.ttype == sqlparse.tokens.Whitespace


def tokenize_sql(sql_exp, schema):
    sql_exp = sql_exp.replace('"', "'")
    if sql_exp.count("'") % 2 != 0:  # odd number of single quotes, meaning the value is incomplete or value contains a single quote
        sql_exp = sql_exp.rstrip(";")
        parse = sqlparse.parse(sql_exp)
        sql = parse[0]
        flat_tokens = sql.flatten()
        sql_tokens = [
            token.value for token in flat_tokens if not _is_whitespace(token)
        ]
        sql_lower = ' '.join(sql_tokens)
        sql_lower = sql_lower.replace(' . ', '.')
        for op in AGG_OPS:
            sql_lower = sql_lower.replace(f" {op} (", f" {op}(")
        sql_lower = sql_lower.replace('( ', '(')
        sql_lower = sql_lower.replace(' )', ')')
        sql_lower = sql_lower.replace(' ,', ',')
        sql_lower = sql_lower.rstrip(";")
        sql_lower += ';'
        mentions = {
            "columns": [],
            "tables": [],
            "keywords": [],
            "values": []
        }
        print(sql_exp, sql_tokens, mentions)
        return sql_tokens, sql_lower, mentions

    sql_exp, values = delexical(sql_exp)
    sql_exp = sql_exp.lower()
    sql_exp = sql_exp.rstrip(";")
    parse = sqlparse.parse(sql_exp)
    sql = parse[0]
    flat_tokens = sql.flatten()
    sql_tokens = [
        token.value for token in flat_tokens if not _is_whitespace(token)
    ]
    mentions = {
        "columns": set(),
        "tables": set(),
        "keywords": set(),
        "values": set([value[1:-1] for value in values.values()]),
    }

    sql_lower = ' '.join(sql_tokens)
    sql_lower = sql_lower.replace(' . ', '.')
    for op in AGG_OPS:
        sql_lower = sql_lower.replace(f" {op} (", f" {op}(")
    sql_lower = sql_lower.replace('( ', '(')
    sql_lower = sql_lower.replace(' )', ')')
    sql_lower = sql_lower.replace(' ,', ',')
    sql_lower = sql_lower.rstrip(";")
    sql_lower += ';'

    for i, tok in enumerate(sql_tokens):
        if tok in SQL_KEYWORDS:
            mentions["keywords"].add(tok)
        if tok in schema["table_names_original"]:
            mentions["tables"].add(tok)
        if is_number(tok):
            mentions["values"].add(convert_to_number(tok))

    for i, tok in enumerate(sql_tokens):
        if tok in schema["column_names_original"]:
            col = tok
            mentions["columns"].add(tok)

    sql_tokens = lexical(sql_tokens, values)
    sql_lower = lexical(sql_lower, values)
    mentions["columns"] = list(mentions["columns"])
    mentions["tables"] = list(mentions["tables"])
    mentions["keywords"] = list(mentions["keywords"])
    mentions["values"] = list(mentions["values"])

    return sql_tokens, sql_lower, mentions


def petershaw_tokenize_sql(sql_exp):
    sql_exp = sql_exp.lower()
    sql_exp = sql_exp.rstrip(";")
    parse = sqlparse.parse(sql_exp)
    sql = parse[0]
    flat_tokens = sql.flatten()
    sql_tokens = [
        token.value for token in flat_tokens if not _is_whitespace(token)
    ]
    return sql_tokens


def is_number(token):
    """Check if token is a SQL number literal."""
    # Note that Python's is_numeric() will return False for values like 30.3.
    try:
        float(token)
        return True
    except ValueError:
        return False


def convert_to_number(token):
    if '.' in token:
        number = float(token)
    else:
        number = int(token)
    return number


petershaw_PLACEHOLDER = "___"


def get_petershaw_template(target):
    """Anonymize quoted substrings and numbers in SQL."""
    # First, replace any numeric token.
    tokens = petershaw_tokenize_sql(target)
    template_tokens = []
    for token in tokens:
        if is_number(token):
            template_tokens.append(petershaw_PLACEHOLDER)
        else:
            template_tokens.append(token)
    template = " ".join(template_tokens)

    # Second, replace any subspan surrounded by single or double quotes.
    in_quotes = False
    quote_token = None
    new_template = ""
    for char in template:
        if in_quotes:
            if char == quote_token:
                in_quotes = False
                quote_token = None
        else:
            if char in ("'", "\""):
                in_quotes = True
                quote_token = char
                new_template += petershaw_PLACEHOLDER
            else:
                new_template += char
    return new_template




def find_random_examples(test_q, questions, split="template", deduplicate_demo="nlq"):
    assert split in ["sql", "nlq", "template", None]
    assert deduplicate_demo in ["sql", "nlq", "template"]
    # questions_shuffled = copy.deepcopy(questions)
    # random.shuffle(questions_shuffled)
    questions_shuffled = random.sample(questions, len(questions))

    seen = set()
    new_questions = []
    for q in questions_shuffled:
        if (split == "nlq" and q["question"] == test_q["question"]) \
                or (split == "sql" and q["query"] == test_q["query"]) \
                or (split == "template" and q["sql_template"] == test_q["sql_template"]):
            continue
        if deduplicate_demo == "nlq" and q["question"] not in seen:
            new_questions.append(q)
            seen.add(q["question"])
        elif deduplicate_demo == "sql" and q["query"] not in seen:
            new_questions.append(q)
            seen.add(q["query"])
        elif deduplicate_demo == "template" and q["sql_template"] not in seen:
            new_questions.append(q)
            seen.add(q["sql_template"])
    return new_questions


def find_simsql(test_q, bm25, questions, retrieval_strategy, split="template", deduplicate_demo="nlq"):
    assert split in ["sql", "nlq", "template", None]
    assert deduplicate_demo in ["sql", "nlq", "template"]
    seen = set()
    if retrieval_strategy in ["simsql_pred", "simsql"]:
        doc_scores = bm25.get_scores(test_q["zeroshot"]["mentions"]["columns"] + test_q["zeroshot"]["mentions"]["keywords"]).tolist()
    else:
        raise NotImplementedError

    questions_scores = zip(questions, doc_scores)
    questions_scores = sorted(questions_scores, key=lambda x: x[1], reverse=True)

    questions = [q for q, s in questions_scores]
    new_questions = []
    for q in questions:
        if (split == "nlq" and q["question"] == test_q["question"]) or \
                (split == "sql" and q["query"] == test_q["query"]) or \
                (split == "template" and q["sql_template"] == test_q["sql_template"]):
            continue
        if deduplicate_demo == "nlq" and q["question"] not in seen:
            new_questions.append(q)
            seen.add(q["question"])
        elif deduplicate_demo == "sql" and q["query"] not in seen:
            new_questions.append(q)
            seen.add(q["query"])
        elif deduplicate_demo == "template" and q["sql_template"] not in seen:
            new_questions.append(q)
            seen.add(q["sql_template"])
    return new_questions

def find_covsql(test_q, bm25, questions, retrieval_strategy="covsql", K=5, split="template", deduplicate_demo="nlq"):
    assert split in ["sql", "nlq", "template", None]
    questions_set = copy.deepcopy(questions)
    used_documents = set()
    for idx, q in enumerate(questions_set):
        if (split == "nlq" and q["question"] == test_q["question"]) \
                or (split == "sql" and q["query"] == test_q["query"]) \
                or (split == "template" and q["sql_template"] == test_q["sql_template"]):
            used_documents.add(q["question"])

    retrieved_questions = []
    if K < len(retrieved_questions):
        K = len(retrieved_questions)

    while len(retrieved_questions) < K:
        if len(questions_set) == len(retrieved_questions):  # no more questions to retrieve
            break
        if retrieval_strategy == "covsql":
            uncover_toks = test_q["zeroshot"]["mentions"]["columns"] + test_q["zeroshot"]["mentions"]["keywords"]

        num_retrieved_questions = len(retrieved_questions)
        while len(uncover_toks) > 0:
            doc_scores = bm25.get_scores(uncover_toks).tolist()

            max_score_index = -1
            max_score = float('-inf')
            for idx, score in enumerate(doc_scores):
                q = questions_set[idx]
                if deduplicate_demo == "nlq" and q["question"] in [x["question"] for x in retrieved_questions]:
                    continue
                if deduplicate_demo == "query" and q["query"] in [x["query"] for x in retrieved_questions]:
                    continue
                if deduplicate_demo == "template" and q["sql_template"] in [x["sql_template"] for x in retrieved_questions]:
                    continue
                if score > max_score and questions_set[idx]["question"] not in used_documents:
                    max_score = score
                    max_score_index = idx
            if max_score == 0 or max_score_index == -1:
                break

            used_documents.add(questions_set[max_score_index]["question"])
            best_q = questions_set[max_score_index]

            if retrieval_strategy =="covsql":
                for col in best_q["gold"]["mentions"]["columns"] + best_q["gold"]["mentions"]["keywords"]:
                    if col in uncover_toks:
                        uncover_toks.remove(col)
    
            retrieved_questions.append(best_q)
            if len(retrieved_questions) == K:
                break
        if num_retrieved_questions == len(retrieved_questions):  # no more new questions in this iteration
            if len(questions_set) == used_documents:  # no more questions to retrieve
                break
            else:
                random.shuffle(questions_set)
                for idx, q in enumerate(questions_set):
                    if q not in retrieved_questions and q["question"] not in used_documents:
                        retrieved_questions.append(q)
                        used_documents.add(q["question"])
                    if len(retrieved_questions) == K:
                        break
                break
    return retrieved_questions