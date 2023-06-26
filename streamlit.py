from PIL import Image
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import pandas as pd
import pydeck as pdk
import pickle
from nlp import preprocess_text
import seaborn as sns
import streamlit as sl
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

sl.set_page_config(layout = 'wide')

#téléchargement du dataframe en cache
@sl.cache_data
def load_df():    
    df1 = pd.read_pickle('df_final_pretraite_fr_1.pickle')
    df2 = pd.read_pickle('df_final_pretraite_fr_2.pickle')
    df3 = pd.read_pickle('df_final_pretraite_fr_3.pickle')
    df4 = pd.read_pickle('df_final_pretraite_fr_4.pickle')
    df = pd.concat([df1, df2, df3, df4])
    return df

#téléchargement du vectorizer
#def load_vectorizer():
#    with open('vectorizer.pkl', 'rb') as file:
#        vectorizer = pickle.load(file)
#    return vectorizer

#téléchargement du modèle ML
#def load_model():
#    with open('mdl.pkl', 'rb') as file:
#        model = pickle.load(file)
#    return model

def open_website(link):
    webbrowser.open(link)

def set_tooltip(nom_etablissement, site_web):
    map_main.deck_widget.tooltip = {
        'html': '<b>Nom : </b> {nom_etablissement} <br/> <b>Type d\'établissement : </b> {category} <br/> <b>Adresse : </b> {rue}, {code_postal} {ville} <br/> <b>Téléphone : </b> {telephone} <br/> <b>Email : </b> {email} <br/> <b>Site web : </b> <a href="{site_web}" target="_blank">{site_web}</a><br/>',
        'style': {
            'color': 'white'
        }
    }
    if site_web:
        open_website(site_web)

#entraînement du modèle ML
@sl.cache_data
def load_ml():
	#df = df.dropna(subset = ['description_pretraitee', 'category'])
	#df.reset_index(drop=True, inplace=True)

	X = df['description_pretraitee']
	y = df['category']

	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

	vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2), min_df=5)

	X_train_vectorized = vectorizer.fit_transform(X_train)

	model = LogisticRegression(class_weight='balanced', max_iter=1000)

	model.fit(X_train_vectorized, y_train)

	return vectorizer, model

df = load_df()

#model = load_model()

#vectorizer = load_vectorizer()

#contenu dans le sidebar
with sl.sidebar:

	#logo
	logo = Image.open('logo.png')
	sl.image(logo, width = 100)

	#filtres

	#catégorie
	type_options = ['Hébergement', 'Restauration']
	user_type = sl.multiselect("Type d'établissement", type_options)

	#mobilité
	mobility = sl.checkbox('Accès mobilité réduite')

	#région
	region_options = df['region'].drop_duplicates().to_list()
	#mettre en première position l'IDF dans la liste des options
	region_options.remove('Île-de-France')
	region_options.insert(0,'Île-de-France')
	user_region = sl.selectbox("Région", region_options)

	#departement
	dep_options = df.loc[df['region'] == user_region]['departement'].drop_duplicates().to_list()
	if user_region == 'Île-de-France':
		dep_options.remove('Paris')
		dep_options.insert(0,'Paris')
	user_dep = sl.selectbox("Département", dep_options)

	#ville
	ville_options = df.loc[df['departement'] == user_dep]['ville'].drop_duplicates().to_list()
	if user_dep == 'Paris':
		ville_options.remove('Paris')
		ville_options.insert(0,'Paris')
	user_ville = sl.selectbox("Ville", ville_options)

sl.header("Bienvenue chez Wild Travelers")

tab1, tab2, tab3 = sl.tabs(['Home', 'Dataviz', 'Robot ML'])

#tab Home
with tab1:

	df_coordinates = df.loc[(df['region'] == user_region) & (df['departement'] == user_dep) & 
		(df['ville'] == user_ville)]

	#df_coordinates = df.loc[df['ville'] == user_ville]

	if bool(user_type) == True:
		df_coordinates = df_coordinates.loc[df_coordinates['category'].isin(user_type)]

	if bool(mobility) == True:
		df_coordinates = df_coordinates.loc[df_coordinates['reducedMobilityAccess'] == True]

	df_coordinates = df_coordinates[['nom_etablissement', 'category', 'latitude', 'longitude', 'rue', 
		'code_postal', 'ville', 'telephone', 'email', 'site_web']]

	if len(df_coordinates) > 0:
		
		sl.write(f"Nombre d'établissements trouvés : {len(df_coordinates)}")

		lat_default = df_coordinates['latitude'].mean()
		lon_default = df_coordinates['longitude'].mean()

	else:
		
		sl.write("Aucun établissement trouvé")

		lat_default = 48.856578
		lon_default = 2.351828

	#cartographie

	#icon_url = 'https://cdn-icons-png.flaticon.com/512/3082/3082383.png'

	icon_hotel_url = "https://raw.githubusercontent.com/Binn1908/Wild-Travelers/main/icon_hotel.png"
	icon_resto_url = "https://raw.githubusercontent.com/Binn1908/Wild-Travelers/main/icon_resto.png"
	
	icon_hotel_data = {
		'url': icon_hotel_url,
		'width': 242,
		'height': 242,
		'anchorY': 242
	}

	icon_resto_data = {
		'url': icon_resto_url,
		'width': 242,
		'height': 242,
		'anchorY': 242
	}

	df_coordinates['icon_data'] = None
	for i in df_coordinates.index:
		if df_coordinates.loc[i,'category'] == 'Hébergement':
			df_coordinates['icon_data'][i] = icon_hotel_data
		else:
			df_coordinates['icon_data'][i] = icon_resto_data

	map_main = sl.pydeck_chart(pdk.Deck(
	    map_style='road',
	    initial_view_state=pdk.ViewState(
	        latitude=lat_default,
	        longitude=lon_default,
	        zoom=12,
	        pitch=40,
	    ),
    	layers=[
        	pdk.Layer(
    			type="IconLayer",
    			data=df_coordinates,
    			get_icon="icon_data",
    			get_size=1,
    			size_scale=15,
   				get_position=["longitude", "latitude"],
    			pickable=True,
			onClick="setTooltip({nom_etablissement}, {site_web})"
        	),
    	],
    	tooltip={
	        'html': '<b>Nom : </b> {nom_etablissement} <br/> <b>Type d\'établissement : </b> {category} <br/> <b>Adresse : </b> {rue}, {code_postal} {ville} <br/> <b>Téléphone : </b> {telephone} <br/> <b>Email : </b> {email} <br/> <b>Site web : </b> {site_web} <br/>',
	        'style': {
	        	'color': 'white'}}
	))

        	#pdk.Layer(
            		#'ScatterplotLayer',
            		#data=df_coordinates,
            		#get_position='[longitude, latitude]',
            		#get_color="category == 'Hébergement' ? [200, 30, 0, 160] : [0, 100, 200, 160]",
            		#get_radius=40,
            		#pickable=True,
            		#tooltip=True

#tab Dataviz
with tab2:

	df_dataviz = df
	
	user_type2 = sl.multiselect("Filtrer par catégorie", type_options)
	
	if user_type2:
		df_dataviz = df_dataviz.loc[df_dataviz['category'].isin(user_type2)]
		
	#KPI 1
	sl.markdown("<h3 style='text-align: center;'>Nombre d'établissements par région et catégorie</h3>", unsafe_allow_html=True)

	establishments_per_region_category = df_dataviz.groupby(['region', 'category']).size().unstack(fill_value=0)

	fig, ax = plt.subplots(figsize = (8,4))
	ax1 = plt.subplot()
	ax1 = sns.heatmap(establishments_per_region_category, annot = True, fmt = 'd', cmap = 'YlGnBu')
	plt.xlabel('Catégorie')
	plt.ylabel('Région')
	#plt.title("Nombre d'établissements par région et catégorie")
	plt.xticks(rotation = 45)
	plt.yticks(rotation = 0)
	plt.tight_layout()
	sl.pyplot(fig)

	#KPI 2
	sl.markdown("<h3 style='text-align: center;'>Distribution des établissements par catégorie</h3>", unsafe_allow_html=True)

	category_counts = df_dataviz['category'].value_counts()
	colors = sns.color_palette('pastel')[0:len(category_counts)]

	fig, ax = plt.subplots(figsize = (4,4))
	ax1 = plt.subplot()
	ax1 = plt.pie(category_counts, labels = category_counts.index, autopct = '%1.1f%%', startangle = 90,
		colors = colors, wedgeprops = {'edgecolor': 'white'})
	plt.axis('equal')
	#plt.title('Distribution des établissements par catégorie')
	plt.legend(title = 'Catégories', loc = 'best', bbox_to_anchor = (1, 0.5))
	plt.gca().add_artist(plt.Circle((0, 0), 0.7, color = 'white'))
	sl.pyplot(fig)

	#KPI 3
	sl.markdown("<h3 style='text-align: center;'>Répartition des établissements avec accès réduit</h3>", unsafe_allow_html=True)

	access_counts = df_dataviz['reducedMobilityAccess'].value_counts()
	labels = access_counts.index
	sizes = access_counts.values
	colors = ['#AAD7AA', '#EEB1B2']
	explode = (0.1, 0) # Écart des tranches
	centre_circle = plt.Circle((0, 0), 0.65, fc='white')

	fig, ax = plt.subplots(figsize = (4,2))
	ax1 = plt.subplot()
	ax1 = plt.pie(sizes, explode = explode, labels = labels, colors = colors, autopct = '%1.1f%%',
		startangle = 90)
	fig = plt.gcf()
	fig.gca().add_artist(centre_circle)
	#plt.title("Répartition des établissements avec accès réduit")
	plt.axis('equal')
	plt.tight_layout()
	sl.pyplot(fig)

	#KPI 4
	sl.markdown("<h3 style='text-align: center;'>Top 10 des villes avec le plus d'établissements</h3>", unsafe_allow_html=True)

	accommodation_counts = df_dataviz['ville'].value_counts()
	#accommodation_counts = df[df['category'] == 'Hébergement']['ville'].value_counts()
	top_10_accommodation_cities = accommodation_counts.nlargest(10)
	colors = sns.color_palette('pastel', len(top_10_accommodation_cities))

	fig, ax = plt.subplots(figsize = (6,2))
	ax1 = plt.subplot()
	ax1 = sns.barplot(x = top_10_accommodation_cities.values, y = top_10_accommodation_cities.index,
		palette = colors)
	#plt.title("Top 10 des villes avec le plus d'hébergements")
	plt.xlabel("Nombre d'établissements")
	plt.ylabel('Ville')
	for i, v in enumerate(top_10_accommodation_cities.values):
	    plt.text(v + 0.2, i, str(v), color = 'black', fontweight = 'bold')
	sl.pyplot(fig)
	
#tab Robot ML
with tab3:

	sl.markdown("<h3 style='text-align: center;'>Test du modèle</h3>", unsafe_allow_html=True)

	example_text = "Nous vous proposons une cuisine moderne avec une belle vue des montagnes."
	new_text = sl.text_area("Veuillez renseigner une description :", value = example_text)

	preprocessed_text = preprocess_text(new_text)

	vectorizer, model = load_ml()

	new_text_vectorized = vectorizer.transform([preprocessed_text])

	predicted_category = model.predict(new_text_vectorized)

	sl.write(f"Prédiction du type d'établissement : {predicted_category[0]}")

#footer
sl.divider()
sl.caption('**Wild Travelers** - un projet à la [Wild Code School](https://www.wildcodeschool.com/)')
