import os
import json
import sqlite3
import subprocess
import subprocess


def get_prompt_length(prompt):
    result = subprocess.run(["node", "codex_prompt_length.mjs", prompt], stdout=subprocess.PIPE)
    prompt_len = eval(result.stdout)
    return prompt_len


def is_number(token):
    """Check if token is a SQL number literal."""
    # Note that Python's is_numeric() will return False for values like 30.3.
    try:
        float(token)
        return True
    except ValueError:
        return False

prompt_length_by_db = {'department_management': 414, 'farm': 770, 'student_assessment': 1701, 'bike_1': 1294, 'book_2': 289, 'musical': 355, 'twitter_1': 459,
                       'product_catalog': 1058, 'flight_1': 574, 'allergy_1': 383, 'store_1': 2421, 'journal_committee': 354,
                       'customers_card_transactions': 1013, 'race_track': 311, 'coffee_shop': 577, 'chinook_1': 2602, 'insurance_fnol': 1010,
                       'medicine_enzyme_interaction': 477, 'university_basketball': 588, 'phone_1': 684, 'match_season': 697, 'climbing': 392,
                       'body_builder': 345, 'election_representative': 340, 'apartment_rentals': 1322, 'game_injury': 582, 'soccer_1': 2144,
                       'performance_attendance': 405, 'college_2': 1878, 'debate': 434, 'insurance_and_eClaims': 1265, 'customers_and_invoices': 2017,
                       'wedding': 442, 'theme_gallery': 403, 'epinions_1': 441, 'riding_club': 771, 'gymnast': 424, 'small_bank_1': 260, 'browser_web': 414,
                       'wrestler': 364, 'school_finance': 583, 'protein_institute': 718, 'cinema': 544, 'products_for_hire': 1713, 'phone_market': 414,
                       'gas_company': 553, 'party_people': 661, 'pilot_record': 591, 'cre_Doc_Control_Systems': 1603, 'company_1': 809,
                       'local_govt_in_alabama': 446, 'formula_1': 2484, 'machine_repair': 596, 'entrepreneur': 338, 'perpetrator': 363, 'csu_1': 633,
                       'candidate_poll': 428, 'movie_1': 284, 'county_public_safety': 488, 'inn_1': 412, 'local_govt_mdm': 1051, 'party_host': 418,
                       'storm_record': 414, 'election': 606, 'news_report': 413, 'restaurant_1': 641, 'customer_deliveries': 2246, 'icfp_1': 450,
                       'sakila_1': 3172, 'loan_1': 541, 'behavior_monitoring': 2470, 'assets_maintenance': 2608, 'station_weather': 606, 'college_1': 1532,
                       'sports_competition': 780, 'manufacturer': 433, 'hr_1': 1420, 'music_1': 815, 'baseball_1': 7197, 'mountain_photos': 525,
                       'program_share': 592, 'e_learning': 1360, 'insurance_policies': 936, 'hospital_1': 2716, 'ship_mission': 339, 'student_1': 223,
                       'company_employee': 502, 'film_rank': 467, 'cre_Doc_Tracking_DB': 1695, 'club_1': 452, 'tracking_grants_for_research': 1733,
                       'network_2': 235, 'decoration_competition': 398, 'document_management': 1315, 'company_office': 487, 'solvency_ii': 1391,
                       'entertainment_awards': 407, 'customers_campaigns_ecommerce': 1735, 'college_3': 1167, 'department_store': 2592, 'aircraft': 1124,
                       'local_govt_and_lot': 2129, 'school_player': 783, 'store_product': 772, 'soccer_2': 379, 'device': 459,
                       'cre_Drama_Workshop_Groups': 4045, 'music_2': 544, 'manufactory_1': 277, 'tracking_software_problems': 1158, 'shop_membership': 575,
                       'voter_2': 541, 'products_gen_characteristics': 940, 'swimming': 739, 'railway': 661, 'customers_and_products_contacts': 1306,
                       'dorm_1': 595, 'customer_complaints': 1176, 'workshop_paper': 373, 'tracking_share_transactions': 868, 'cre_Theme_park': 2283,
                       'game_1': 554, 'customers_and_addresses': 1287, 'music_4': 523, 'roller_coaster': 377, 'ship_1': 330, 'city_record': 792,
                       'e_government': 2096, 'school_bus': 429, 'flight_company': 574, 'cre_Docs_and_Epenses': 957, 'scientist_1': 328, 'wine_1': 499,
                       'train_station': 470, 'driving_school': 1572, 'activity_1': 651, 'flight_4': 944, 'tracking_orders': 1027, 'architecture': 497,
                       'culture_company': 589, 'concert_singer': 604, 'pets_1': 379, 'car_1': 656, 'flight_2': 375, 'employee_hire_evaluation': 566,
                       'cre_Doc_Template_Mgt': 657, 'course_teach': 360, 'museum_visit': 393, 'wta_1': 1068, 'battle_death': 546,
                       'student_transcripts_tracking': 2005, 'tvshow': 725, 'poker_player': 355, 'voter_1': 403, 'world_1': 843, 'orchestra': 700,
                       'network_1': 312, 'dog_kennels': 1872, 'singer': 332, 'real_estate_properties': 1342}

OOD_SCHEMA_MAXLEN = 1000


def get_prompt_length(dataset, db_id):
    prompt = generate_create_table_prompt(dataset, db_id, prompt_db="CreateTableSelectCol",limit_value=3)
    result = subprocess.run(["node", "codex_prompt_length.mjs", prompt], stdout=subprocess.PIPE)
    prompt_len = eval(result.stdout)
    return prompt_len



def normalize_create_table(table_name, create_table_statement, tblcol_to_description=None):
    create_table_statement = create_table_statement.strip()
    create_table_statement = create_table_statement.replace("`", "\"").replace("'", "\"").replace("[", "\"").replace("]", "\"")
    create_table_statement = create_table_statement.replace("\"", '')
    create_table_statement = create_table_statement.replace('\t', ' ').replace('\n', ' ')
    create_table_statement = ' '.join(create_table_statement.split())
    create_table_statement_split = [""]
    num_left = 0
    for tok in create_table_statement:
        if tok == "(":
            num_left += 1
            create_table_statement_split[-1] += tok
        elif tok == ")":
            num_left -= 1
            create_table_statement_split[-1] += tok
        elif tok != ',':
            create_table_statement_split[-1] += tok
        if tok == ',':
            if num_left == 1:
                create_table_statement_split.append("")
                continue
            else:
                create_table_statement_split[-1] += tok
                continue
    # create_table_statement = create_table_statement.split(',')
    create_table_statement = create_table_statement_split
    new_create_table_statement = []
    for i, x in enumerate(create_table_statement):
        if i == 0:
            x = x.split('(')
            x1 = x[0].strip()
            x2 = ','.join(x[1:]).strip()
            new_create_table_statement.append(x1 + " (")
            new_create_table_statement.append(x2 + ",")
        elif i == len(create_table_statement) - 1:
            x = x.split(')')
            x1 = ')'.join(x[:-1]).strip()
            x2 = x[-1].strip()
            new_create_table_statement.append(x1)
            new_create_table_statement.append(x2 + ")")
        else:
            new_create_table_statement.append(x.strip() + ",")
    if tblcol_to_description is not None:
        # print(tblcol_to_description)
        new_create_table_statement_with_desc = []
        for row in new_create_table_statement:
            col = row.split(',')[0].split(' ')[0].strip().replace("\"", "").lower()
            tblcol = table_name.lower() + '.' + col
            # print(tblcol)
            if tblcol in tblcol_to_description:
                row += f" -- {tblcol_to_description[tblcol]}"
            new_create_table_statement_with_desc.append(row)
        new_create_table_statement = new_create_table_statement_with_desc
    return '\n'.join(new_create_table_statement)



def extract_create_table_prompt_column_example(db_id, db_path, table_path=None, limit_value=3, add_column_description=False):
    if add_column_description:
        with open(table_path, "r") as f:
            schemas = json.load(f)
            for schema in schemas:
                if schema["db_id"] == db_id:
                    break
            column_names_original = schema["column_names_original"]
            column_descriptions = schema["column_descriptions"]
            table_names_original = schema["table_names_original"]
        tblcol_to_description = {}
        for col, desc in zip(column_names_original, column_descriptions):
            if col[0] == -1:
                continue
            tbl_col = table_names_original[col[0]] + '.' + col[1]
            tbl_col = tbl_col.lower()
            tblcol_to_description[tbl_col] = desc
    else:
        tblcol_to_description = None

    lower_case = True
    table_query = "SELECT * FROM sqlite_master WHERE type='table';"
    tables = sqlite3.connect(db_path).cursor().execute(table_query).fetchall()
    prompt = ""
    for table in tables:
        table_name = table[1]
        if lower_case:
            table_name = table_name.lower()
        create_table_statement = table[-1]

        table_info_query = f"PRAGMA table_info({table_name});"
        top_k_row_query = f"SELECT * FROM {table_name} LIMIT {limit_value};"
        headers = [x[1] for x in sqlite3.connect(db_path).cursor().execute(table_info_query).fetchall()]
        if lower_case:
            create_table_statement = normalize_create_table(table_name, create_table_statement, tblcol_to_description)
            create_table_statement = create_table_statement.lower()
            headers = [x.lower() for x in headers]

        prompt += create_table_statement + ";\n"
        if limit_value > 0:
            prompt_columns = []
            for col_name in headers:
                if col_name.lower() == "index":
                    top_k_rows = list(range(limit_value))
                    top_k_rows = '    '.join([str(x) for x in top_k_rows])
                else:
                    top_k_row_query = f"SELECT distinct \"{col_name}\" FROM {table_name} LIMIT {limit_value + 10};"
                    top_k_rows = sqlite3.connect(db_path).cursor().execute(top_k_row_query).fetchall()
                    top_k_rows = [x[0].strip() if isinstance(x[0], str) else x[0] for x in top_k_rows]  # remove \n and space prefix and suffix in cell value
                    top_k_rows = [x if x is not None else "" for x in top_k_rows]
                    top_k_rows = ', '.join([str(x) if is_number(x) else '"' + str(x) + '"' for x in top_k_rows][:limit_value])
                prompt_columns.append(f"{col_name}: {top_k_rows};")

            prompt += "/*\n"
            prompt += f"Columns in {table_name} and {limit_value} distinct examples in each column:\n"
            prompt += "\n".join(prompt_columns)
            prompt += "\n*/\n"
        prompt += "\n"

    return prompt


def generate_create_table_prompt(dataset, db_id, prompt_db="CreateTableSelectCol",limit_value=3):
    root_path="data"
    db_dir = f"{root_path}/{dataset}/database"
    table_path = f"{root_path}/{dataset}/tables/tables.json"
    
    db_path = os.path.join(db_dir, db_id, db_id + ".sqlite")
    if prompt_db == "CreateTableSelectCol_description" and dataset in ["spider-train", "spider"]:
        prompt_db = "CreateTableSelectCol"
    
    if prompt_db == "CreateTableSelectCol":
        db_prompt = extract_create_table_prompt_column_example(db_id, db_path, limit_value=limit_value)
    elif prompt_db == "CreateTableSelectCol_description":
        db_prompt = extract_create_table_prompt_column_example(db_id, db_path, table_path, limit_value=limit_value, add_column_description=True)
    else:
        print(prompt_db)
        raise NotImplementedError


    prompt = db_prompt + "-- Using valid SQLite, answer the following questions for the tables provided above.\n"

    return (prompt)

