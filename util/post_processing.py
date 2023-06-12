def extract_data_from_tables(tables):
    result = {}

    # birth date
    result['birth_date'] = f"{tables[0][1]['content']} {tables[0][2]['content']} {tables[0][3]['content']}"

    # items table
    result['items'] = {}
    for cell in tables[1]:
        key = 'row_' + str(cell['row']).zfill(2)
        if cell['row'] in (3, 5, 7, 9, 11, 13):
            if result['items'].get(key) is None: result['items'][key] = {} # fist time needs to initialize the dict             
            if cell['column'] == 0: result['items'][key]['code'] = cell['content']
            elif cell['column'] == 11: result['items'][key]['provider_id'] = cell['content']            
        elif cell['row'] in (4, 6, 8, 10, 12, 14):
            if result['items'].get(key) is None: result['items'][key] = {} # fist time needs to initialize the dict 
            if cell['column'] == 0: result['items'][key]['date_from'] = cell['content']
            elif cell['column'] == 1: result['items'][key]['date_to'] = cell['content']
            elif cell['column'] == 2: result['items'][key]['place_of_service'] = cell['content']
            elif cell['column'] == 3: result['items'][key]['emg'] = cell['content']            
            elif cell['column'] == 4: result['items'][key]['cpt'] = cell['content']
            elif cell['column'] == 5: result['items'][key]['modifier'] = cell['content']            
            elif cell['column'] == 6: result['items'][key]['diagnosis'] = cell['content']
            elif cell['column'] == 7: result['items'][key]['charges'] = cell['content']
            elif cell['column'] == 8: result['items'][key]['units'] = cell['content']
            elif cell['column'] == 11: result['items'][key]['provider_id'] = cell['content']

    # charge table (if exists)
    if len(tables) > 2:
        for cell in tables[2]:
            if cell['row'] == 0:
                if cell['column'] == 0: result['tax_id'] = cell['content']
                elif cell['column'] == 1: result['account_number'] = cell['content']
                elif cell['column'] == 2: result['total_charge'] = cell['content']
                elif cell['column'] == 3: result['amount_paid'] = cell['content']
    
    return result