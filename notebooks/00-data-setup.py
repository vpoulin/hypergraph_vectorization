import numpy as np
import pandas as pd
import matplotlib
import seaborn as sns
import csv

def read_format_citeseer():
    content = pd.read_csv('../data/citeseer-doc-classification/citeseer.content', sep='\t', header=None)
    cites = pd.read_csv('../data/citeseer-doc-classification/citeseer.cites', sep='\t', header=None)
    n = content.shape[1]
    
    labels = {str(content.loc[i, 0]): str(content.loc[i, n-1]) for i in range(content.shape[0])}
    doc_word_matrix = content[[i+1 for i in range(n-2)]].to_numpy()
    citations = cites.groupby(0).aggregate(lambda x: list(x)).reset_index().rename(columns={0:'paper', 1:'citation'})
    
    citation_labels = [labels[x] if x in labels.keys() else 'No_label' for x in citations.paper]
    doc_labels = list(content[n-1])
    
    color_key =  {'AI': '#0b559f',
     'Agents': '#61409b',
     'DB': '#d41020',
     'HCI': '#008080',
     'IR': '#955072',
     'ML': '#4cb063',
     'No_label': '#777777bb'}

    return(citations, citation_labels, doc_word_matrix, doc_labels, color_key)


def read_format_recipes(recipe_min_size=3):
    ingredients_id = pd.read_csv('../data/cat-edge-Cooking/node-labels.txt', sep='\t', header=None)
    ingredients_id.index = [x+1 for x in ingredients_id.index]
    ingredients_id.columns = ['Ingredient']
    
    recipes_with_id = []
    with open('../data/cat-edge-Cooking/hyperedges.txt', newline = '') as hyperedges:
        hyperedge_reader = csv.reader(hyperedges, delimiter='\t')
        for hyperedge in hyperedge_reader:
            recipes_with_id.append(hyperedge)
            
    recipes_all = [[ingredients_id.loc[int(i)]['Ingredient'] for i in x] for x in recipes_with_id]
    
    # Keep recipes with 3 ingredients and more
    keep_recipes = np.where(np.array([len(x) for x in recipes_all])>=recipe_min_size)[0]
    recipes = [recipes_all[i] for i in keep_recipes]
    
    recipes_label_id_all = pd.read_csv('../data/cat-edge-Cooking/hyperedge-labels.txt', sep='\t', header=None)
    recipes_label_id_all.columns = ['label']
    recipes_label_id = recipes_label_id_all.iloc[keep_recipes].reset_index()

    label_name = pd.read_csv('../data/cat-edge-Cooking/hyperedge-label-identities.txt', sep='\t', header=None)
    label_name.columns = ['country']
    label_name.index = [x+1 for x in label_name.index]
    
    grps_tmp = {
        'asian' : ('chinese', 'filipino', 'japanese','korean', 'thai', 'vietnamese'),
        'american' : ('brazilian', 'mexican', 'southern_us'),
        'english' : ('british', 'irish'),
        'islands' : ('cajun_creole', 'jamaican'),
        'europe' : ('french', 'italian', 'spanish'),
        'others' : ('greek', 'indian', 'moroccan', 'russian')
    }

    grps = {key:[key+'.'+x for x in value] for key, value in grps_tmp.items()}


    color_key = {}
    for l, c in zip(grps['asian'], sns.color_palette("Blues", 6)[0:]):
        color_key[l] = matplotlib.colors.rgb2hex(c)
    for l, c in zip(grps['american'], sns.color_palette("Purples", 4)[1:]):
        color_key[l] = matplotlib.colors.rgb2hex(c)
    for l, c in zip(grps['others'], sns.color_palette("YlOrRd", 4)):
        color_key[l] = matplotlib.colors.rgb2hex(c)
    for l, c in zip(grps['europe'], sns.color_palette("light:teal", 4)[1:]):
        color_key[l] = matplotlib.colors.rgb2hex(c)
    for l, c in zip(grps['islands'], sns.color_palette("light:#660033", 4)[1:3]):
        color_key[l] = matplotlib.colors.rgb2hex(c)
    for l, c in zip(grps['english'], sns.color_palette("YlGn", 4)[1:]):
        color_key[l] = matplotlib.colors.rgb2hex(c)
    color_key["ingredient"] = "#777777bb"
    
    new_names = []
    for key, value in grps.items():
        new_names = new_names + value

    label_name['new_label'] = [new_name for x in label_name.country for new_name in new_names if x in new_name]
    
    return(recipes, recipes_label_id, ingredients_id, label_name, color_key)